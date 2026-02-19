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
from eureka_ml_insights.metrics import CountAggregator
from eureka_ml_insights.metrics.docvqa_metrics import OCRBenchSubstringMetric

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


class OCRBENCH_PIPELINE(ExperimentConfig):
    """
    ExperimentConfig pipeline for the OCRBench benchmark (v1).
    1,000 questions across 10 OCR task types.
    Dataset: echo840/OCRBench on HuggingFace.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "echo840/OCRBench",
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"answer": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/ocrbench_templates/default.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

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

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(OCRBenchSubstringMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["OCRBenchSubstringMetric_result"],
                        "normalize": True,
                        "filename_base": "OCRBench_Accuracy",
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["OCRBenchSubstringMetric_result"],
                        "normalize": True,
                        "group_by": "question_type",
                        "filename_base": "OCRBench_Accuracy_By_QuestionType",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [self.data_processing_comp, self.inference_comp, self.evalreporting_comp],
            self.log_dir,
        )


class OCRBENCH_V2_PIPELINE(ExperimentConfig):
    """
    ExperimentConfig pipeline for the OCRBench v2 benchmark.
    10,000 questions across 30 OCR task types in 31 scenarios.
    Dataset: lmms-lab/OCRBench-v2 on HuggingFace.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:

        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lmms-lab/OCRBench-v2",
                    "split": "test",
                    "transform": SequenceTransform(
                        [
                            ColumnRename(name_mapping={"answers": "ground_truth"}),
                        ]
                    ),
                },
            ),
            prompt_template_path=os.path.join(
                os.path.dirname(__file__),
                "../prompt_templates/ocrbench_templates/default.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

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

        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(OCRBenchSubstringMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["OCRBenchSubstringMetric_result"],
                        "normalize": True,
                        "filename_base": "OCRBenchV2_Accuracy",
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["OCRBenchSubstringMetric_result"],
                        "normalize": True,
                        "group_by": "type",
                        "filename_base": "OCRBenchV2_Accuracy_By_Type",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [self.data_processing_comp, self.inference_comp, self.evalreporting_comp],
            self.log_dir,
        )
