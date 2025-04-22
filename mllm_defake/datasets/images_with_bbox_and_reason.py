from abc import abstractmethod
from pathlib import Path

from mllm_defake.datasets.images_only import RealFakeDataset


class BboxReasonDataset(RealFakeDataset):
    @abstractmethod
    def get_bbox_reason(self, image_path: Path) -> list[tuple[tuple[int, int, int, int], str]]:
        """
        This method returns the bounding boxes and reasons for the fake images.

        @param image_path: The path of the image file.
        @return: A list of bounding boxes and reasons. Each bounding box is represented as a tuple of (x1, y1, x2, y2), and the reason is a string.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class WhydFake(BboxReasonDataset):
    """
    This dataset class is designed for the WhydFake dataset intended for researchers. The format is specialized for the WhydFake dataset.

    ./WhydFake
    ├── images
    │   ├── batch_a
    │   ├── batch_b
    │   ├── batch_c
    │   ├── real_batch
    │   └── real_batch_a
    ├── scripts
    └── annotations.jsonl
    """

    def __init__(self, data_root: str | Path):
        """
        @param data_root: The WhydFake root folder. `images` folder should be under this folder.
        """
        super().__init__()
        self.data_root = Path(data_root)
