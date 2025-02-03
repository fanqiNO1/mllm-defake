import base64
import imghdr
import json
from pathlib import Path


def read(p: Path | str, bytes: bool = False) -> str | bytes:
    """
    Read the contents of a file.
    """
    return Path(p).read_text() if not bytes else Path(p).read_bytes()


def pprint_messages_payload(msgs: list) -> str:
    """
    Pretty-print the messages payload.
    """
    ret = "\n=== BEGINNING OF CONVERSATION ===\n"
    turn_count = 0
    for i, msg in enumerate(msgs):
        role = msg["role"]
        if role == "system" or role == "developer":
            ret += f"{role.capitalize() + ' ' * (12 - len(role))}: {json.dumps(msg['content'])}\n"
        else:
            turn_count += 1
            ret += f"{role.capitalize() + ' ' * (12 - len(role))}: {json.dumps(msg['content']) if len(json.dumps(msg['content'])) < 100 else json.dumps(msg['content'])[:100] + '...'}\n"
    ret += f"=== END OF CONVERSATION (Turns: {turn_count}) ==="
    return ret


def encode_image_to_base64(image_path):
    """
    Encodes an image from the given file path to base64 format.

    Args:
        image_path: The path to the image file.

    Returns:
        A string in the format "data:image/{image_type};base64,{base64_encoded_image}"
    """

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_type = imghdr.what(image_path)
        if image_type:
            base64_encoded_image = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/{image_type};base64,{base64_encoded_image}"
        else:
            raise ValueError("Unsupported image format")
