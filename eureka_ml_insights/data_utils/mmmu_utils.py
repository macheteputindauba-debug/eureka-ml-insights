import json
from dataclasses import dataclass, field

import pandas as pd

from .transform import DFTransformBase

MMMUCategories = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

MMMUTaskToCategories = {task: cat[0] for cat in MMMUCategories.items() for task in cat[1]}

MMMUAll = [task for cat in MMMUCategories.values() for task in cat]


@dataclass
class CreateMMMUPrompts(DFTransformBase):
    """
    Create prompts in the MMMU format for mutiple-choice and open-ended questions
    The original code is located in https://github.com/MMMU-Benchmark/MMMU/tree/main/eval/utils and has
    small modifications to work in this framework.
    """

    def __init__(self):
        self.multi_choice_example_format = "{}\n{}\nAnswer with the option's letter from the given choices directly."
        self.short_ans_example_format = "{}\nAnswer the question using a single word or phrase."

    def _create_prompt(self, sample):
        question = sample["question"]
        options = sample["options"]
        example = ""
        if sample["question_type"] == "multiple-choice":
            start_chr = "A"
            prediction_range = []
            index2ans = {}
            for option in options:
                prediction_range.append(start_chr)
                example += f"({start_chr}) {option}\n"
                index2ans[start_chr] = option
                start_chr = chr(ord(start_chr) + 1)

            prompt = self.multi_choice_example_format.format(question, example)
        else:
            prompt = self.short_ans_example_format.format(question)

        return prompt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)

        return df

@dataclass
class CreateMUIRBENCHPrompts(DFTransformBase):
    """
    Create prompts in the MUIRBENCH format for mutiple-choice and open-ended questions
    The original code is located in https://github.com/muirbench/MuirBench/tree/main/eval/utils and has
    small modifications to work in this framework.
    """

    def _create_prompt(self, sample, use_hint=True):
        question = sample['question']
        choices = sample['options']

        # Question
        question_text = f"Question: {question}"
        
        # Choices
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A')+i)}) {choice}")
        choices_text = "\n".join(texts)

        # Hint
        if use_hint:
            hint_text = f"Hint: Please provide the correct option letter, such as A, B, C, D, directly."
        else:
            hint_text = ""

        # Answer Prefix
        prompt = "Answer:"
        
        # Full Prompt
        elements = [question_text, choices_text, hint_text, prompt]
        query = "\n".join([e for e in elements if e != ""])
        query = query.strip()

        return query

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)

        return df    

tasks_exist = ['person_reid', 'multiple_image_captioning', 'spot_the_similarity', 'face_retrieval', 'sketch2image_retrieval', 'handwritten_retrieval', 'spot_the_diff', 'image2image_retrieval', 'vehicle_retrieval', 'text2image_retrieval',
'general_action_recognition', 'video_captioning', 'next_img_prediction', 'temporal_ordering', 'meme_vedio_understanding', 'action_quality_assessment', 'temporal_localization', 'mevis',
'ravens_progressive_matrices', 'threed_indoor_recognition', 'point_tracking', 'threed_cad_recognition', 'single_object_tracking']

@dataclass
class CreateMMIUPrompts(DFTransformBase):
    """
    Create prompts in the MMUI format for mutiple-choice and open-ended questions
    The original code is located in https://github.com/OpenGVLab/MMIU/blob/main and has
    small modifications to work in this framework.
    """

    def _create_prompt(self, sample, use_hint=True):
        question = sample['question']
        options = sample['options']
        context = sample["context"]
                    
        if sample['task'] in tasks_exist:
            question = question + '\n' + context
        else:
            question = context + '\n' + question

        question = question + '\nPlease answer the option directly like A,B,C,D...'

        return question

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)

        return df


BLINK_SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


@dataclass
class CreateBLINKPrompts(DFTransformBase):
    """
    Create prompts for the BLINK benchmark.
    Combines the task context (prompt column) with the question and formatted choices.
    Also normalizes the answer column from "(A)" to "A".
    """

    def _create_prompt(self, sample):
        context = sample.get("prompt", "")
        question = sample["question"]
        choices = sample["choices"]

        # Format choices
        choices_text = ""
        for i, choice in enumerate(choices):
            choices_text += f"({chr(ord('A') + i)}) {choice}\n"

        # Build full prompt
        parts = []
        if context:
            parts.append(context)
        parts.append(question)
        parts.append(choices_text)
        parts.append("Answer with the option's letter from the given choices directly.")

        return "\n".join(parts)

    @staticmethod
    def _normalize_answer(answer):
        """Strip parentheses from answer: '(A)' -> 'A'."""
        if isinstance(answer, str):
            return answer.strip().strip("()")
        return answer

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df["prompt"] = df.apply(self._create_prompt, axis=1)
        df["answer"] = df["answer"].apply(self._normalize_answer)
        return df


@dataclass
class MergeBaselineAnswers(DFTransformBase):
    """Merge baseline model answers from a JSONL file into the dataframe.

    The baseline file should have one JSON object per line with at least
    'question_id' and 'output' fields (WildVision format).
    """

    baseline_path: str = ""
    key_column: str = "question_id"
    output_column: str = "baseline_output"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        baseline = {}
        with open(self.baseline_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                baseline[entry[self.key_column]] = entry["output"]
        df[self.output_column] = df[self.key_column].map(baseline)
        return df