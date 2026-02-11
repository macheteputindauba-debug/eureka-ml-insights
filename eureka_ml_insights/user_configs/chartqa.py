"""ChartQA benchmark pipeline configuration.

ChartQA is a benchmark for chart question answering that tests a model's ability
to understand charts and answer questions about them.

Evaluation follows the lmms-eval approach:
- Prompt: "{question} Answer the question with a single word or number."
- Metric: Relaxed accuracy (5% tolerance for numbers, case-insensitive for text)

References:
- ChartQA: https://github.com/vis-nlp/ChartQA
- lmms-eval: https://github.com/EvolvingLMMs-Lab/lmms-eval

Dataset: https://huggingface.co/datasets/lmms-lab/ChartQA
"""

import os
from typing import Any

from eureka_ml_insights.configs import (
    AggregatorConfig,
    DataSetConfig,
    EvalReportingConfig,
    ExperimentConfig,
    InferenceConfig,
    MetricConfig,
    ModelConfig,
    PipelineConfig,
    PromptProcessingConfig,
)
from eureka_ml_insights.core import EvalReporting, Inference, PromptProcessing
from eureka_ml_insights.data_utils import (
    ColumnRename,
    DataReader,
    HFDataReader,
    MMDataLoader,
    SequenceTransform,
)
from eureka_ml_insights.metrics import CountAggregator
from eureka_ml_insights.metrics.chartqa_metrics import ChartQARelaxedAccuracyMetric


class CHARTQA_PIPELINE(ExperimentConfig):
    """ChartQA benchmark pipeline using the lmms-lab/ChartQA dataset.

    This pipeline evaluates models on chart question answering using relaxed accuracy:
    - 5% tolerance for numerical answers
    - Case-insensitive exact match for text answers

    Uses lmms-eval style prompting for reproducibility.
    """

    def configure_pipeline(
        self, model_config: ModelConfig, resume_from: str = None, **kwargs: dict[str, Any]
    ) -> PipelineConfig:
        # Data processing component with prompt template
        self.data_processing_comp = PromptProcessingConfig(
            component_type=PromptProcessing,
            data_reader_config=DataSetConfig(
                HFDataReader,
                {
                    "path": "lmms-lab/ChartQA",
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
                "../prompt_templates/chartqa_templates/default.jinja",
            ),
            output_dir=os.path.join(self.log_dir, "data_processing_output"),
        )

        # Inference component
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

        # Evaluation and reporting component
        self.evalreporting_comp = EvalReportingConfig(
            component_type=EvalReporting,
            data_reader_config=DataSetConfig(
                DataReader,
                {
                    "path": os.path.join(self.inference_comp.output_dir, "inference_result.jsonl"),
                    "format": ".jsonl",
                },
            ),
            metric_config=MetricConfig(ChartQARelaxedAccuracyMetric),
            aggregator_configs=[
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ChartQARelaxedAccuracyMetric_result"],
                        "normalize": True,
                        "filename_base": "ChartQA_Accuracy",
                    },
                ),
                AggregatorConfig(
                    CountAggregator,
                    {
                        "column_names": ["ChartQARelaxedAccuracyMetric_result"],
                        "group_by": "type",
                        "normalize": True,
                        "filename_base": "ChartQA_Accuracy_By_Type",
                    },
                ),
            ],
            output_dir=os.path.join(self.log_dir, "eval_report"),
        )

        return PipelineConfig(
            [
                self.data_processing_comp,
                self.inference_comp,
                self.evalreporting_comp,
            ],
            self.log_dir,
        )
