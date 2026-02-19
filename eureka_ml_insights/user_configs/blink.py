import os
from typing import Any

from eureka_ml_insights.configs.experiment_config import ExperimentConfig
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    AddColumnAndData,
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.data_utils.mmmu_utils import BLINK_SUBTASKS, CreateBLINKPrompts
from eureka_ml_insights.metrics import CountAggregator, MMMUMetric

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


class BLINK_BASELINE_PIPELINE(ExperimentConfig):
    """
    ExperimentConfig pipeline for the BLINK benchmark.
    Evaluates visual perception across 14 subtasks with multiple-choice questions.
    Dataset: BLINK-Benchmark/BLINK on HuggingFace.
    """

    def configure_pipeline(self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]) -> PipelineConfig:

        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "BLINK-Benchmark/BLINK",
                    "tasks": BLINK_SUBTASKS,
                    "split": "val",
                    "transform": SequenceTransform(
                        [
                            CreateBLINKPrompts(),
                            ColumnRename(name_mapping={"answer": "ground_truth", "choices": "target_options"}),
                            AddColumnAndData(column_name="question_type", data="multiple-choice"),
                        ]
                    ),
                },
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
            ignore_failure=False,
        )

        # Configure the inference component
        self.inference_comp = InferenceConfig(
            component_type=Inference,
            model_config=model_config,
            data_loader_config=DataSetConfig(
                MMDataLoader,
                {
                    "path": os.path.join(self.data_processing_comp.output_dir, "transformed_data.jsonl"),
                    "image_column_names": ["image_1", "image_2", "image_3", "image_4"],
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
            metric_config=MetricConfig(MMMUMetric),
            aggregator_configs=[
                AggregatorConfig(CountAggregator, {"column_names": ["MMMUMetric_result"], "normalize": True}),
                AggregatorConfig(
                    CountAggregator,
                    {"column_names": ["MMMUMetric_result"], "group_by": "sub_task", "normalize": True},
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig([self.data_processing_comp, self.inference_comp, self.evalreporting_comp], self.log_dir)
