import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

SUPPORTED_BASIC_CLASSIFIERS = ["canny", "comfor"]


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BasicClassifier(ABC):
    """
    Abstract base class for lightweight image classifiers used as baselines.

    This class provides a framework for simple, heuristic, or traditional ML-based
    image classifiers that can be used to establish baseline performance for
    fake image detection.

    Subclasses must implement the `classify` method to provide specific detection logic.
    """

    def __init__(self) -> None:  # noqa: B027 (allow empty __init__)
        pass

    @abstractmethod
    def classify(self, sample: Path, label: int | bool) -> int:
        """
        Classify a single image as real or fake.

        Parameters
        ----------
        sample : Path
            Path to the image file to classify.
        label : int or bool
            Known label for the sample (may be used for debugging).
            - 1 or True: real image
            - 0 or False: fake image

        Returns
        -------
        int
            Classification result:
            - 1: real image
            - 0: fake image
        """
        pass

    @staticmethod
    def _update_metrics(y_true, y_pred, pbar) -> None:
        """
        Update progress bar with current evaluation metrics.

        Parameters
        ----------
        y_true : list
            List of true labels (1 for real, 0 for fake).
        y_pred : list
            List of predicted labels (1 for real, 0 for fake).
        pbar : tqdm
            Progress bar to update.
        """
        if y_pred:
            acc = accuracy_score(y_true, y_pred) * 100
            real_acc = (
                len([1 for i, y in enumerate(y_true) if y == 1 and y_pred[i] == 1]) / max(1, y_true.count(1)) * 100
            )
            fake_acc = (
                len([1 for i, y in enumerate(y_true) if y == 0 and y_pred[i] == 0]) / max(1, y_true.count(0)) * 100
            )
            pbar.set_postfix(
                all=f"{acc:.2f}%",
                reals=f"{real_acc:.2f}%",
                fakes=f"{fake_acc:.2f}%",
            )

    def evaluate(
        self,
        real_samples: list[Path],
        fake_samples: list[Path],
        output_path: Path,
        continue_from: pd.DataFrame | None = None,
    ) -> tuple[float, float, float]:
        """
        Evaluate the classifier on a set of real and fake images.

        Parameters
        ----------
        real_samples : list[Path]
            List of paths to real image samples.
        fake_samples : list[Path]
            List of paths to fake image samples.
        output_path : Path
            Path to save the evaluation results CSV file.
        continue_from : pd.DataFrame, optional
            If provided, continue evaluation from this previously saved results dataframe.

        Returns
        -------
        tuple[float, float, float]
            A tuple of (accuracy, precision, recall) metrics.

        Notes
        -----
        Results are written to a CSV file with columns:
        - path: Path to the image
        - label: True label (1 for real, 0 for fake)
        - pred: Predicted label (1 for real, 0 for fake)

        Raises
        ------
        ValueError
            If input sample lists contain invalid types.
        """
        if not all(isinstance(p, Path) for p in real_samples):
            raise ValueError("`real_samples` must be a list[Path].")
        if not all(isinstance(p, Path) for p in fake_samples):
            raise ValueError("`fake_samples` must be a list[Path].")

        samples: list[tuple[Path, int]] = [(p, 1) for p in real_samples] + [(p, 0) for p in fake_samples]

        if continue_from is not None:
            processed = set(continue_from["path"].apply(Path))
            samples = [(p, y) for p, y in samples if p not in processed]
            write_mode = "a"
        else:
            continue_from = pd.DataFrame(columns=["path", "label", "pred"])
            write_mode = "w"

        y_true, y_pred = [], []
        pbar = tqdm(enumerate(samples), total=len(samples), desc="Evaluating...")
        for i, (sample, label) in pbar:
            pbar.set_description(f"Eval {sample.name[:19]}")
            try:
                pred = self.classify(sample, label)
                y_true.append(label)
                y_pred.append(pred)

                pd.DataFrame({"path": [sample], "label": [label], "pred": [pred]}).to_csv(
                    output_path,
                    index=False,
                    mode="w" if (i == 0 and write_mode == "w") else "a",
                    header=(i == 0 and write_mode == "w"),
                )

                self._update_metrics(y_true, y_pred, pbar)

            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error processing {sample.name}: {exc}")

        if not y_pred:  # no successful predictions
            print("No valid predictions produced.")
            return 0.0, 0.0, 0.0

        # Final metrics (avoid div/0 with 1e‑10)
        accuracy = sum(t == p for t, p in zip(y_true, y_pred, strict=False)) / len(y_true)
        precision = sum(t == p == 1 for t, p in zip(y_true, y_pred, strict=False)) / (
            sum(p == 1 for p in y_pred) + 1e-10
        )
        recall = sum(t == p == 1 for t, p in zip(y_true, y_pred, strict=False)) / (sum(t == 1 for t in y_true) + 1e-10)
        return accuracy, precision, recall


# --------------------------------------------------------------------------- #
# Canny‑edge baseline
# --------------------------------------------------------------------------- #
class CannyClassifier(BasicClassifier):
    """
    Edge-density baseline detector using Canny edge filtering.

    This classifier uses the density of edges detected by the Canny algorithm
    to differentiate between real and synthetic images, based on the observation
    that AI-generated images often have different edge patterns than real photos.
    """

    @staticmethod
    def _canny_density(sample: Path) -> float:
        """
        Calculate the edge density of an image using Canny edge detection.

        Parameters
        ----------
        sample : Path
            Path to the image file.

        Returns
        -------
        float
            The density of edges (ratio of edge pixels to total pixels).
        """
        img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        return np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    def classify(self, sample: Path, label: int | bool = -1) -> int:
        """
        Classify an image based on its edge density.

        Parameters
        ----------
        sample : Path
            Path to the image file to classify.
        label : int or bool
            Known label for the sample. Unused but reserved for interface.

        Returns
        -------
        int
            Classification result:
            - 1: real image (edge density > threshold)
            - 0: fake image (edge density <= threshold)
        """
        return 1 if self._canny_density(sample) > 0.0672 else 0


# --------------------------------------------------------------------------- #
# Community‑Forensics ViT baseline
# --------------------------------------------------------------------------- #
class ComForClassifier(BasicClassifier):
    """
    Wrapper for the Community-Forensics ViT detector.

    This classifier uses a Vision Transformer model trained on the Community Forensics
    dataset to detect AI-generated images. The model was developed by Park et al.
    and is available at https://jespark.net/projects/2024/community_forensics/.
    """

    def __init__(
        self,
        community_forensics_checkpoint_path: Path,
        input_size: Literal["224", "384"] = "384",
        device: str | torch.device | None = None,
    ) -> None:
        """
        Initialize the Community-Forensics classifier.

        Parameters
        ----------
        community_forensics_checkpoint_path : Path
            Path to the pre-trained model checkpoint file.
        input_size : Literal["224", "384"], default="384"
            Input resolution for the model (224x224 or 384x384).
        device : str or torch.device, default="cuda:0"
            Device to run the model on (e.g., "cuda:0", "cpu").

        Raises
        ------
        ValueError
            If input_size is not "224" or "384".
        ImportError
            If required dependencies are missing.
        """
        super().__init__()
        self.check_dependencies()

        import torch.nn as nn
        import timm
        from torchvision.transforms import (
            CenterCrop,
            Compose,
            ConvertImageDtype,
            Normalize,
            Resize,
            ToTensor,
        )

        class _ComForModel(nn.Module):
            def __init__(self, in_size: int, dev: str | torch.device):
                super().__init__()
                self.in_size = in_size
                self.dev = dev

                vit_name = (
                    "vit_small_patch16_224.augreg_in21k_ft_in1k"
                    if in_size == 224
                    else "vit_small_patch16_384.augreg_in21k_ft_in1k"
                )
                self.vit = timm.create_model(vit_name, pretrained=False)
                self.vit.head = nn.Linear(384, 1, bias=True, device=dev)

                self.preprocess = Compose(
                    [
                        Resize(256 if in_size == 224 else 440),
                        CenterCrop(in_size),
                        ToTensor(),
                        Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711],
                        ),
                        ConvertImageDtype(torch.float32),
                    ]
                )

            def forward(self, pil_img: Image.Image):
                x = self.preprocess(pil_img).unsqueeze(0).to(self.dev)
                return torch.sigmoid(self.vit(x))

        sz = 224 if input_size == "224" else 384 if input_size == "384" else None
        if sz is None:
            raise ValueError("`input_size` must be '224' or '384'.")

        self.model = _ComForModel(sz, device).to(device).eval()
        state = torch.load(community_forensics_checkpoint_path, map_location=device)
        self.model.load_state_dict(state["model"], strict=True)

    # .............................................................
    @staticmethod
    def check_dependencies() -> None:
        """
        Check if required dependencies are installed.

        Raises
        ------
        ImportError
            If any required dependency is missing.
        """
        missing = [m for m in ("torch", "torchvision", "timm") if importlib.util.find_spec(m) is None]
        if missing:
            raise ImportError(f"Missing dependencies: {', '.join(missing)}")

    # .............................................................
    def _fake_probability(self, sample: Path) -> float:
        """
        Calculate the probability that an image is fake.

        Parameters
        ----------
        sample : Path
            Path to the image file.

        Returns
        -------
        float
            Probability score between 0 and 1, where higher values indicate
            higher likelihood of being a fake image.
        """
        img = Image.open(sample).convert("RGB")
        # model returns prob(fake); invert for prob(real)
        return 1.0 - self.model(img).item()

    # .............................................................
    def classify(self, sample: Path, label: int | bool) -> int:
        """
        Classify an image as real or fake using the Community-Forensics model.

        Parameters
        ----------
        sample : Path
            Path to the image file to classify.
        label : int or bool
            Known label for the sample (unused but required by the interface).

        Returns
        -------
        int
            Classification result:
            - 1: real image (probability > 0.5)
            - 0: fake image (probability <= 0.5)
        """
        # Threshold chosen from original paper/README recommendations
        return 1 if self._fake_probability(sample) > 0.5 else 0
