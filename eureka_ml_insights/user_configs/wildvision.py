import os
import re
from typing import Any

from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    AddColumn,
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MapStringsTransform,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.mmmu_utils import MergeBaselineAnswers
from eureka_ml_insights.metrics import AverageAggregator

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG as PERSONAL_GPT4O


def _parse_reward(text):
    """Parse pairwise verdict from judge output into a numeric reward score.

    Assistant A = baseline (Claude-3-Sonnet), Assistant B = model under test.
    Returns numeric reward from the model's perspective, matching VLMEvalKit:
      A>>B = -2, A>B = -1, A=B = 0, B>A = 1, B>>A = 2
    Returns None for unparseable outputs.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r"\[\[([AB<>=]+)\]\]", text)
    if not match:
        return None
    verdict = match.group(1)
    reward_map = {
        "A>>B": -2,
        "A>B": -1,
        "A=B": 0,
        "B>A": 1,
        "B>>A": 2,
    }
    return reward_map.get(verdict, None)


def _parse_win(text):
    """Parse pairwise verdict into win (1) or not-win (0) for win rate calculation."""
    reward = _parse_reward(text)
    if reward is None:
        return None
    return 1 if reward > 0 else 0


class WILDVISION_PIPELINE(ExperimentConfig):
    """
    ExperimentConfig pipeline for the WildVision-Bench benchmark.
    Evaluates vision-language models on 500 real-world image+instruction pairs
    using GPT-4o as a pairwise judge against Claude-3-Sonnet baseline.
    Dataset: WildVision/wildvision-bench (vision_bench_0617) on HuggingFace.

    Requires baseline_answers_path kwarg pointing to the Claude-3-Sonnet JSONL
    from https://github.com/WildVision-AI/WildVision-Bench/tree/main/data/vision_bench_0617/model_answers
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        LLM_JUDGE_CONFIG = kwargs.get("llm_judge_config", PERSONAL_GPT4O)
        baseline_answers_path = kwargs.get("baseline_answers_path")
        if not baseline_answers_path:
            raise ValueError(
                "baseline_answers_path is required. Download claude-3-sonnet-20240229.jsonl from "
                "https://github.com/WildVision-AI/WildVision-Bench/tree/main/data/vision_bench_0617/model_answers"
            )

        # Data processing: load WildVision data
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "WildVision/wildvision-bench",
                    "tasks": "vision_bench_0617",
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            CopyColumn(column_name_src="instruction", column_name_dst="prompt"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Model inference on image + instruction
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["image"],
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # Prepare judge prompt: merge baseline answers, rename model_output -> response
        self.eval_data_pre_processing = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            MergeBaselineAnswers(baseline_path=baseline_answers_path),
                            ColumnRename(name_mapping={"model_output": "response"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/wildvision_templates/scoring_prompt.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "eval_data_pre_processing_output"),
        )

        # Judge inference: GPT-4o pairwise comparison (with images)
        self.eval_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=LLM_JUDGE_CONFIG,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.eval_data_pre_processing.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["image"],
                },
            ),
            output_dir=os.path.join(self.log_dir, "eval_inference_result"),
        )

        # Parse verdicts into numeric reward scores and win flags
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.eval_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": SequenceTransform(
                        [
                            AddColumn(column_name="reward"),
                            CopyColumn(column_name_src="model_output", column_name_dst="reward"),
                            MapStringsTransform(columns=["reward"], mapping=_parse_reward),
                            AddColumn(column_name="win"),
                            CopyColumn(column_name_src="model_output", column_name_dst="win"),
                            MapStringsTransform(columns=["win"], mapping=_parse_win),
                        ]
                    ),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["reward"],
                        "filename_base": "WildVision_Reward",
                        "ignore_non_numeric": True,
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["win"],
                        "filename_base": "WildVision_WinRate",
                        "ignore_non_numeric": True,
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.eval_data_pre_processing,
                self.eval_inference_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
