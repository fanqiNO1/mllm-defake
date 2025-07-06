import json
import re

import imagesize


SPECIAL_TOKENS = {
    "internvl3": {
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
        str: The model type, either "internvl3" or "qwen2.5-vl".
    """
    for model_type, special_tokens in SPECIAL_TOKENS.items():
        for token in special_tokens.values():
            if token in completion:
                return model_type
    raise ValueError("Model type not recognized in the completion text.")


def _post_process_bbox(bbox: list[float], images: list[str], model_type: str) -> list[float]:
    """Post-process the bounding box coordinates based on the model type.

    Args:
        bbox (list[float]): The bounding box coordinates.
        model_type (str): The model type, either "internvl3" or "qwen2.5-vl".

    Returns:
        list[float]: The post-processed bounding box coordinates.
    """
    if model_type == "qwen2.5-vl":
        return bbox
    elif model_type == "internvl3":
        # since intervl3 uses `norm1000`, we need to de-normalize the bounding box
        if len(images) != 1:
            raise ValueError("For internvl3, only one image is expected.")
        width, height = imagesize.get(images[0])
        x1, y1, x2, y2 = bbox
        x1, y1 = x1 * width / 1000, y1 * height / 1000
        x2, y2 = x2 * width / 1000, y2 * height / 1000
        return [x1, y1, x2, y2]
    else:
        raise ValueError(f"Model type {model_type} is not supported for post-processing bounding boxes.")


def cal_iou(completion: str, objects: dict, images: list[str]) -> float:
    """Calculate the IoU (Intersection over Union) between the completion and the objects.

    Args:
        completion (str): The completion text.
        objects (dict): The objects to compare against, containing "ref" and "bbox".
        images (list[str]): List of image paths corresponding to the completion.

    Returns:
        float: The IoU value.
    """
    # since objects is empty, there is no need to perform grounding
    if not objects:
        return 1.0
    # extract bounding boxes from the completion
    model_type = _get_model_type(completion)
    ref_object_start = SPECIAL_TOKENS[model_type]["ref_object_start"]
    ref_object_end = SPECIAL_TOKENS[model_type]["ref_object_end"]
    box_start = SPECIAL_TOKENS[model_type]["box_start"]
    box_end = SPECIAL_TOKENS[model_type]["box_end"]
    pattern = rf"{ref_object_start}.*?{ref_object_end}\s*{box_start}(.*?){box_end}"
    bboxes = []
    for box in re.findall(pattern, completion):
        # qwen2.5-vl: [[x1, y1], [x2, y2]]
        # internvl3: (x1,y1),(x2,y2)
        box = box.strip().replace(" ", "").replace("[[", "[").replace("]]", "]")
        box = box.replace("(", "[").replace(")", "]")
        # converted to [x1, y1], [x2, y2]
        try:
            x1, y1, x2, y2 = json.loads([box])[0]
            bbox = _post_process_bbox([x1, y1, x2, y2], images, model_type)
            bboxes.append(bbox)
        except Exception:
            continue
    # calculate IoU
    raise NotImplementedError("I have no idea")


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
    matches = re.search(pattern, completion)
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
