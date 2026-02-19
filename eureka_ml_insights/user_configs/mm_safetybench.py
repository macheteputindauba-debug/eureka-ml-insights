"""MM-SafetyBench pipeline configurations.

MM-SafetyBench is a benchmark for evaluating the safety of Multimodal Large Language
Models (MLLMs) against image-based manipulations. It contains 13 safety scenarios
with 5,040 text-image pairs.

Reference: https://github.com/isXinLiu/MM-SafetyBench
Paper: https://arxiv.org/abs/2311.17600
Dataset: https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench
"""

import os
from typing import Any

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
from eureka_ml_insights.configs.model_configs import OAI_GPT4_1106_PREVIEW_CONFIG as DEFAULT_LLM_JUDGE
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    CopyColumn,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics.reports import AverageAggregator

# All available scenario subsets in MM-SafetyBench
MM_SAFETYBENCH_SCENARIOS = [
    "Illegal_Activity",
    "HateSpeech",
    "Malware_Generation",
    "Physical_Harm",
    "EconomicHarm",
    "Fraud",
    "Sex",
    "Political_Lobbying",
    "Privacy_Violence",
    "Legal_Opinion",
    "Financial_Advice",
    "Health_Consultation",
    "Gov_Decision",
]

# Available image types/splits
MM_SAFETYBENCH_SPLITS = ["SD", "SD_TYPO", "TYPO", "Text_only"]


class MM_SAFETYBENCH_PIPELINE(ExperimentConfig):
    """MM-SafetyBench evaluation pipeline.

    Evaluates MLLM safety against query-relevant image manipulations.
    Uses GPT-4 as an LLM judge to classify responses as safe/unsafe.

    Args (via kwargs):
        scenario: One of the 13 safety scenarios (default: "Illegal_Activity")
        image_type: One of "SD", "SD_TYPO", "TYPO", "Text_only" (default: "SD")
        llm_judge_config: Model config for the LLM judge (default: GPT-4)

    Metric:
        Attack Success Rate (ASR) = unsafe responses / total responses
        Safety Rate = 1 - ASR = safe responses / total responses
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Get configuration options
        scenario = kwargs.get("scenario", "Illegal_Activity")
        image_type = kwargs.get("image_type", "SD")
        llm_judge_config = kwargs.get("llm_judge_config", DEFAULT_LLM_JUDGE)

        # Validate inputs
        if scenario not in MM_SAFETYBENCH_SCENARIOS:
            raise ValueError(f"Invalid scenario: {scenario}. Must be one of {MM_SAFETYBENCH_SCENARIOS}")
        if image_type not in MM_SAFETYBENCH_SPLITS:
            raise ValueError(f"Invalid image_type: {image_type}. Must be one of {MM_SAFETYBENCH_SPLITS}")

        # Data processing component
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "PKU-Alignment/MM-SafetyBench",
                    "name": scenario,
                    "split": image_type,
                    "transform": SequenceTransform(
                        [
                            CopyColumn(column_name_src="question", column_name_dst="prompt"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Inference component
        image_columns = ["image"] if image_type != "Text_only" else []
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "image_column_names": image_columns,
                    "load_images": image_type != "Text_only",
                },
            ),
            output_dir=os.path.join(self.log_dir, "inference_result"),
            resume_from=resume_from,
        )

        # LLM judge data processing component
        self.judge_data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": ColumnRename(name_mapping={"model_output": "response"}),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__), "../prompt_templates/mm_safetybench_templates/safety_judge.jinja"
            ),
            output_dir=os.path.join(self.log_dir, "judge_data_processing_output"),
        )

        # LLM judge inference component
        self.judge_inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=llm_judge_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.judge_data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "load_images": False,
                },
            ),
            max_concurrent=500,
            output_dir=os.path.join(self.log_dir, "judge_inference_result"),
        )

        # Evaluation and reporting component
        # Score of 1 = safe, 0 = unsafe
        # Average score = safety rate (1 - attack success rate)
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.judge_inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                    "transform": ColumnRename(name_mapping={"model_output": "safety_score"}),
                },
            ),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["safety_score"],
                        "filename_base": f"MM_SafetyBench_{scenario}_{image_type}_SafetyRate",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.judge_data_processing_comp,
                self.judge_inference_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )


class MM_SAFETYBENCH_ALL_SCENARIOS_PIPELINE(ExperimentConfig):
    """MM-SafetyBench evaluation across all 13 scenarios for a given image type.

    This pipeline runs evaluation on all safety scenarios and aggregates results.

    Args (via kwargs):
        image_type: One of "SD", "SD_TYPO", "TYPO", "Text_only" (default: "SD")
        llm_judge_config: Model config for the LLM judge (default: GPT-4)
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        image_type = kwargs.get("image_type", "SD")
        llm_judge_config = kwargs.get("llm_judge_config", DEFAULT_LLM_JUDGE)

        if image_type not in MM_SAFETYBENCH_SPLITS:
            raise ValueError(f"Invalid image_type: {image_type}. Must be one of {MM_SAFETYBENCH_SPLITS}")

        components = []

        # Create pipeline components for each scenario
        for i, scenario in enumerate(MM_SAFETYBENCH_SCENARIOS):
            scenario_dir = os.path.join(self.log_dir, scenario)

            # Data processing
            data_proc = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    HFDataReader,
                    {
                        "path": "PKU-Alignment/MM-SafetyBench",
                        "name": scenario,
                        "split": image_type,
                        "transform": SequenceTransform(
                            [
                                CopyColumn(column_name_src="question", column_name_dst="prompt"),
                            ]
                        ),
                    },
                ),
                output_dir=os.path.join(scenario_dir, "data_processing_output"),
            )
            components.append(data_proc)

            # Model inference
            image_columns = ["image"] if image_type != "Text_only" else []
            inference = InferenceConfig(
                component_type=Inference,
                model_config=model_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(data_proc.output_dir, "transformed_data.jsonl"),
                        "image_column_names": image_columns,
                        "load_images": image_type != "Text_only",
                    },
                ),
                output_dir=os.path.join(scenario_dir, "inference_result"),
                resume_from=resume_from if i == 0 else None,
            )
            components.append(inference)

            # Judge data processing
            judge_data_proc = PromptProcessingConfig(
                component_type=PromptProcessing,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(inference.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": ColumnRename(name_mapping={"model_output": "response"}),
                    },
                ),
                prompt_template_path=os.path.join(
                    os.path.dirname(__file__), "../prompt_templates/mm_safetybench_templates/safety_judge.jinja"
                ),
                output_dir=os.path.join(scenario_dir, "judge_data_processing_output"),
            )
            components.append(judge_data_proc)

            # Judge inference
            judge_inference = InferenceConfig(
                component_type=Inference,
                model_config=llm_judge_config,
                data_loader_config=DataSetConfig(
                    MMDataLoader,
                    {
                        "path": os.path.join(judge_data_proc.output_dir, "transformed_data.jsonl"),
                        "load_images": False,
                    },
                ),
                max_concurrent=500,
                output_dir=os.path.join(scenario_dir, "judge_inference_result"),
            )
            components.append(judge_inference)

            # Per-scenario evaluation
            eval_report = EvalReportingConfig(
                component_type=EvalReporting,
                data_reader_config=DataSetConfig(
                    DataReader,
                    {
                        "path": os.path.join(judge_inference.output_dir, "inference_result.jsonl"),
                        "format": ".jsonl",
                        "transform": ColumnRename(name_mapping={"model_output": "safety_score"}),
                    },
                ),
                aggregator_configs=[
                    AggregatorConfig(
                        AverageAggregator,
                        {
                            "column_names": ["safety_score"],
                            "filename_base": f"MM_SafetyBench_{scenario}_SafetyRate",
                        },
                    ),
                ],
                output_dir=os.path.join(scenario_dir, "eval_report"),
            )
            components.append(eval_report)

        return PipelineConfig(components, self.log_dir)
