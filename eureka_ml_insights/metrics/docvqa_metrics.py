"""DocVQA ANLS (Average Normalized Levenshtein Similarity) metric.

Standard evaluation metric for DocVQA, following the original implementation:
https://rrc.cvc.uab.es/?ch=17&com=tasks

For each prediction:
1. Compute normalized Levenshtein similarity against each ground truth answer
2. Take the maximum similarity
3. If max similarity >= threshold (0.5), score = max similarity; otherwise score = 0.0

The final ANLS score is the average across all questions.

Also includes OCRBenchSubstringMetric for OCRBench evaluation, which checks
if any ground truth answer appears as a case-insensitive substring of the
model output (matching VLMEvalKit's approach).
"""

import ast

from .metrics_base import ClassicMetric


class DocVQAANLSMetric(ClassicMetric):
    """ANLS metric for DocVQA evaluation.

    Handles ground_truth as a list of acceptable answers.
    Returns a float score per row (use AverageAggregator to compute final ANLS).
    """

    def __init__(self, model_output_col: str = "model_output", threshold: float = 0.5):
        super().__init__(model_output_col)
        self.threshold = threshold

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return DocVQAANLSMetric._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    @staticmethod
    def _normalized_levenshtein_similarity(s1: str, s2: str) -> float:
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - DocVQAANLSMetric._levenshtein_distance(s1, s2) / max_len

    @staticmethod
    def _parse_ground_truths(ground_truth) -> list:
        """Parse ground_truth into a list of strings, handling various formats."""
        if isinstance(ground_truth, list):
            return [str(gt) for gt in ground_truth]
        if isinstance(ground_truth, str):
            try:
                parsed = ast.literal_eval(ground_truth)
                if isinstance(parsed, list):
                    return [str(gt) for gt in parsed]
            except (ValueError, SyntaxError):
                pass
            return [ground_truth]
        return [str(ground_truth)]

    def __evaluate__(self, model_output, ground_truth, is_valid):
        if not is_valid:
            return 0.0

        if model_output is None or ground_truth is None:
            return 0.0

        prediction = str(model_output).strip().lower()
        ground_truths = self._parse_ground_truths(ground_truth)

        max_similarity = 0.0
        for gt in ground_truths:
            gt = gt.strip().lower()
            similarity = self._normalized_levenshtein_similarity(prediction, gt)
            max_similarity = max(max_similarity, similarity)

        return max_similarity if max_similarity >= self.threshold else 0.0


class OCRBenchSubstringMetric(ClassicMetric):
    """Substring matching metric for OCRBench evaluation.

    Checks if any ground truth answer appears as a case-insensitive substring
    of the model output. Returns "correct"/"incorrect" for use with CountAggregator.
    Handles ground_truth as a single string or a list of acceptable answers.
    """

    @staticmethod
    def _parse_ground_truths(ground_truth) -> list:
        """Parse ground_truth into a list of strings, handling various formats."""
        if isinstance(ground_truth, list):
            return [str(gt) for gt in ground_truth]
        if isinstance(ground_truth, str):
            try:
                parsed = ast.literal_eval(ground_truth)
                if isinstance(parsed, list):
                    return [str(gt) for gt in parsed]
            except (ValueError, SyntaxError):
                pass
            return [ground_truth]
        return [str(ground_truth)]

    def __evaluate__(self, model_output, ground_truth, is_valid):
        if not is_valid:
            return "none"

        if model_output is None or ground_truth is None:
            return "incorrect"

        prediction = str(model_output).strip().lower()
        ground_truths = self._parse_ground_truths(ground_truth)

        for gt in ground_truths:
            gt = gt.strip().lower()
            if gt in prediction:
                return "correct"

        return "incorrect"
