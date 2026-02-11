"""ChartQA relaxed accuracy metric.

This implementation follows the canonical relaxed accuracy metric from Pix2Struct:
https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py

ChartQA uses a relaxed accuracy metric:
- For numerical answers: correct if within 5% of the ground truth
- For text answers: case-insensitive exact match
- Percentages are normalized (e.g., "50%" and "0.5" are treated as equivalent)
"""

import re

from .metrics_base import ClassicMetric


class ChartQARelaxedAccuracyMetric(ClassicMetric):
    """Relaxed accuracy metric for ChartQA evaluation.

    Follows the Pix2Struct implementation:
    - For numeric answers: correct if within 5% relative change of ground truth
    - For non-numeric answers: case-insensitive exact match
    - Percentages are converted to decimals for comparison (50% -> 0.5)
    """

    def __init__(self, model_output_col: str = "model_output", max_relative_change: float = 0.05):
        super().__init__(model_output_col)
        self.max_relative_change = max_relative_change

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, handling percentages.

        This matches the Pix2Struct implementation exactly.
        """
        if text is None:
            return None
        text = str(text).strip()
        try:
            if text.endswith("%"):
                # Convert percentages to floats (e.g., "50%" -> 0.5)
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    def _extract_answer(self, text: str) -> str:
        """Extract the answer from model output.

        Handles common patterns like:
        - "The answer is X"
        - "X" (direct answer)
        - Strips whitespace and common prefixes
        """
        if text is None:
            return ""
        text = str(text).strip()

        # Remove common answer prefixes
        prefixes = [
            r"^the answer is[:\s]*",
            r"^answer[:\s]*",
            r"^the value is[:\s]*",
            r"^it is[:\s]*",
            r"^approximately[:\s]*",
            r"^about[:\s]*",
            r"^around[:\s]*",
        ]
        for prefix in prefixes:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)

        # Remove trailing punctuation
        text = text.rstrip(".,;:!?")

        return text.strip()

    def _relaxed_correctness(self, target: str, prediction: str) -> bool:
        """Calculate relaxed correctness following Pix2Struct.

        See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
        "We consider an answer to be correct if it is within 5% of the gold answer.
        For non-numeric answers, we still need an exact match."
        """
        prediction_float = self._to_float(prediction)
        target_float = self._to_float(target)

        if prediction_float is not None and target_float is not None and target_float != 0:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
            return relative_change <= self.max_relative_change
        else:
            return prediction.lower() == target.lower()

    def __evaluate__(self, model_output, ground_truth, is_valid):
        if not is_valid:
            return "none"

        if model_output is None or ground_truth is None:
            return "incorrect"

        # Extract answer from model output
        prediction = self._extract_answer(str(model_output))
        target = str(ground_truth).strip()

        # Check relaxed correctness
        if self._relaxed_correctness(target, prediction):
            return "correct"

        return "incorrect"
