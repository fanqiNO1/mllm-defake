SPECIAL_TOKNES = {
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


def cal_iou(completion: str, objects: dict) -> float:
    """Calculate the IoU (Intersection over Union) between the completion and the objects.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        float: The IoU value.
    """
    raise NotImplementedError


def check_format(completion: str) -> bool:
    """Check if the completion format is correct.

    Args:
        completion (str): The completion text.

    Returns:
        bool: True if the format is correct, False otherwise.
    """
    raise NotImplementedError


def check_label(completion: str, objects: dict) -> bool:
    """Check if the completion contains the correct label.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".

    Returns:
        bool: True if the label is correct, False otherwise.
    """
    raise NotImplementedError
