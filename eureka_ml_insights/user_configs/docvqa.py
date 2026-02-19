import os
from typing import Any

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics import AverageAggregator
from eureka_ml_insights.metrics.docvqa_metrics import DocVQAANLSMetric

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)


class DOCVQA_VAL_PIPELINE(ExperimentConfig):
    """
    ExperimentConfig pipeline for the DocVQA benchmark.
    Evaluates document visual question answering using ANLS metric.
    Dataset: lmms-lab/DocVQA on HuggingFace.
    Defaults to the validation split.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lmms-lab/DocVQA",
                    "tasks": "DocVQA",
                    "split": "validation",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"answers": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/docvqa_templates/default.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Configure the inference component
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

        # Configure the evaluation and reporting component
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(DocVQAANLSMetric),
            aggregator_configs=[
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["DocVQAANLSMetric_result"],
                        "filename_base": "DocVQA_ANLS",
                    },
                ),
                AggregatorConfig(
                    AverageAggregator,
                    {
                        "column_names": ["DocVQAANLSMetric_result"],
                        "group_by": "question_types",
                        "filename_base": "DocVQA_ANLS_By_QuestionType",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [self.data_processing_comp, self.inference_comp, self.evalreporting_comp],
            self.log_dir,
        )


class DOCVQA_TEST_PIPELINE(DOCVQA_VAL_PIPELINE):
    """DocVQA pipeline on the test split."""

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]) -> PipelineConfig:
        config = super().configure_pipeline(model_config, resume_from, **kwargs)
        self.data_processing_comp.data_reader_config.init_args["split"] = "test"
        return config
