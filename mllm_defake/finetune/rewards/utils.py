import json
import re


SPECIAL_TOKENS = {
    "internvl2_5": {
        "ref_object_start": "<ref>",
        "ref_object_end": "</ref>",
        "box_start": "<box>",
        "box_end": "</box>",
    },
    "qwen2.5-vl": {
        "ref_object_start": "<|object_ref_start|>",
        "ref_object_end": "<|object_ref_end|>",
        "box_start": "<|box_start|>",
        "box_end": "<|box_end|>",
    },
}


def _get_model_type(completion: str) -> str:
    """Determine the model type based on the completion text.

    Args:
        completion (str): The completion text.

    Returns:
        str: The model type, either "internvl2_5" or "qwen2.5-vl".
    """
    for model_type, special_tokens in SPECIAL_TOKENS.items():
        for token in special_tokens.values():
            if token in completion:
                return model_type
    raise ValueError("Model type not recognized in the completion text.")


def cal_iou(completion: str, objects: dict) -> float:
    """Calculate the IoU (Intersection over Union) between the completion and the objects.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        float: The IoU value.
    """
    # since objects is empty, there is no need to perform grounding
    if not objects:
        return 1.0
    # extract bounding boxes from the completion
    raise NotImplementedError("I have no idea.")


def check_format(completion: str, objects: dict) -> bool:
    """Check if the completion format is correct.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    # since objects is empty, the grounding format is not required
    if not objects:
        pattern = r"<think>.*?</think>\s*<verdict>.*?</verdict>"
    else:
        raise NotImplementedError("I have no idea")
    # check if the completion matches the pattern
    matches = re.search(pattern, completion, re.DOTALL)
    return matches is not None


def check_label(completion: str, objects: dict) -> bool:
    """Check if the completion contains the correct label.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        bool: True if the label is correct, False otherwise.
    """
    # since objects is empty, the ground truth is `real`
    if not objects:
        ground_truth = "real"
    else:
        ground_truth = "generated"
    # extract the label from the completion
    label_match = re.search(r"<verdict>(.*?)</verdict>", completion)
    predicted_label = label_match.group(1) if label_match else ""
    predicted_label = predicted_label.strip().lower()
    # check if the predicted label matches the ground truth
    return predicted_label == ground_truth
