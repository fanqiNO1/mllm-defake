import importlib
import random
import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm

from mllm_defake.utils import encode_image_to_base64
from mllm_defake.vllms import VLLM

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, level="INFO")
# Add . to sys.path to allow importing modules from the current directory


class MLLMClassifier:
    @logger.catch
    def __init__(self, prompt: dict, model: VLLM, random_seed: int = 3809):
        """
        Initialize a multimodal LLM-based classifier for fake image detection.

        Parameters
        ----------
        prompt : dict
            A prompt template dictionary with format_version=3 that contains conversations
            and response variables for the model.
        model : VLLM
            A VLLM model instance with infer_raw method for handling image-text prompts.
        random_seed : int, default=3809
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If prompt format is invalid or model doesn't have required methods.
        """
        if not isinstance(prompt, dict) or "format_version" not in prompt or prompt.get("format_version") != "3":
            raise ValueError(f"Invalid prompt. Expected a v3 JSON object, but got: {prompt}")
        if not isinstance(model, VLLM) or not hasattr(model, "infer_raw"):
            raise ValueError(f"Invalid model. Expected a VLLM object with `infer_raw` method, but got: {model}")
        self.prompt = prompt
        self.model = model
        random.seed(random_seed)  # Set the random seed for reproducibility

    def get_decorator_func(self, decorator: str) -> callable:
        """
        Import and return a decorator function by name.

        Parameters
        ----------
        decorator : str
            The full path to the decorator function in format "module_name.function_name".

        Returns
        -------
        callable
            The imported decorator function.

        Notes
        -----
        Decorators should be callable functions with the signature `def decorator(cache: dict) -> None`
        that modify the cache dictionary during inference.
        """
        module_name, func_name = decorator.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name, package="mllm_defake.decorators")
            return getattr(module, func_name)
        except ImportError:
            try:
                module = importlib.import_module(f"decorators.{module_name}")
                return getattr(module, func_name)
            except ImportError:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, func_name)
                except ImportError as e:
                    logger.error(
                        "Decorator module not found: {}; sys.path = {}",
                        module_name,
                        sys.path,
                    )
                    raise ImportError(f"Decorator module not found: {module_name} ({e})") from e

    def decorate(self, cache: dict, decorator: str) -> None:
        """
        Apply a decorator function to the cache dictionary.

        Parameters
        ----------
        cache : dict
            The cache dictionary to modify.
        decorator : str
            The full path to the decorator function.

        Raises
        ------
        ImportError
            If the decorator module or function cannot be imported.
        """
        decorator_func = self.get_decorator_func(decorator)
        if decorator_func is None:
            raise ImportError(f"The function {decorator} could not be imported")
        decorator_func(cache)

    def llm_query(self, sample: Path, label: int | bool, should_print_response: bool = False) -> str:
        """
        Query the LLM model with an image and expected label.

        Parameters
        ----------
        sample : Path
            Path to the image file to analyze.
        label : int or bool
            Label indicating whether the sample is real or fake:
            - 1 or True: real image
            - 0 or False: fake image
            - -1: unknown
        should_print_response : bool, default=False
            Whether to print the model's response for debugging.

        Returns
        -------
        str
            The raw response from the model.
        """
        # For caching model responses within this classification job, keys should be `response_var_name` values
        cache = {}
        # Prefill cache with sample and label
        cache["_image_path"] = sample
        cache["image_url"] = encode_image_to_base64(sample)
        cache["label"] = (
            "real" if (label == 1 or label is True) else "fake" if (label == 0 or label is False) else "unknown"
        )  # Converts 0/1 to fake/real, and True/False to real/fake

        def replace_string_with_cache(string: str) -> str:
            # Works like f-string, but uses cache values
            return string.format(**cache)

        r = replace_string_with_cache
        for i, conv in enumerate(self.prompt["conversations"]):
            logger.debug(f"Processing conversation #{i} ({conv['id']})")
            messages = []
            # Process decorators
            if "decorators" in conv:
                for decorator in conv["decorators"]:
                    self.decorate(cache, decorator)
            for payload_item in conv["payload"]:
                role, content = payload_item[:2]
                if role == "system":
                    messages.append(
                        {
                            "role": "system",
                            "content": r(content),
                        }
                    )
                elif role == "user":
                    if len(payload_item) >= 3:
                        # The user message contains an image
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": r(content)},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": r(payload_item[2])},
                                    },
                                ],
                            }
                        )
                    else:
                        # The user message contains text
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": r(content)},
                                ],
                            }
                        )
                elif role == "assistant":
                    # The assistant message contains a model response
                    messages.append(
                        {
                            "role": "assistant",
                            "content": r(content),
                        }
                    )
            response = self.model.infer_raw(messages)
            cache[conv["response_var_name"]] = response
            logger.debug(f"Sandbox cache[{conv['response_var_name']}] = {response}")
            if should_print_response:
                logger.info(f"Response to conversation #{i} ({conv['id']}):\n{response}")
        return cache["result"]

    def postprocess(self, result: str) -> int:
        """
        Extract classification decision from the model's text response.

        Parameters
        ----------
        result : str
            The raw text response from the model.

        Returns
        -------
        int
            Classification result:
            - 1: real image
            - 0: fake image
            - -1: undetermined

        Notes
        -----
        This is a simple implementation that searches for keywords in the response.
        It may not work optimally for all models or responses.
        """
        result = result.lower().strip()
        last_real = max(
            result.rfind("real"), result.rfind(" natural"), result.rfind("真实的")
        )  # Include Chinese characters since a lot of models are trained on Chinese data
        last_fake = max(result.rfind("fake"), result.rfind("generated"), result.rfind("生成的"))
        # If both searches fail, we cannot determine the final decision
        if last_real == -1 and last_fake == -1:
            logger.warning("Could not determine the final decision from the response: {}", result)
            return -1
        return 1 if last_real > last_fake else 0

    def classify(self, sample: Path, label: int | bool | None = None, should_print_response: bool = False) -> int:
        """
        Classify a single image as real or fake.

        Parameters
        ----------
        sample : Path
            Path to the image file to classify.
        label : int or bool, default=-1
            Known label for the sample, used for caching and possible prompt engineering.
            - 1 or True: real image
            - 0 or False: fake image
            - -1: unknown
        should_print_response : bool, default=False
            Whether to print the model's response for debugging.

        Returns
        -------
        int
            Classification result:
            - 1: real image
            - 0: fake image
            - -1: undetermined
        """
        if label is None:
            label = -1
        response = self.llm_query(sample, label, should_print_response)
        return self.postprocess(response)

    @staticmethod
    def _update_metrics(y_true: list, y_pred: list, pbar: tqdm) -> None:
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
        continue_from: pd.DataFrame = None,
        should_print_response: bool = False,
    ) -> tuple[float, float, float]:
        """
        Evaluate the model on a set of real and fake images.

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
        should_print_response : bool, default=False
            Whether to print model responses during evaluation.

        Returns
        -------
        tuple[float, float, float]
            A tuple of (accuracy, precision, recall) metrics.

        Notes
        -----
        Results are written to a CSV file with columns:
        - path: Path to the image
        - label: True label (1 for real, 0 for fake)
        - pred: Predicted label (1 for real, 0 for fake, -1 for undetermined)
        - response: Raw model response
        """
        # Save references to samples for later use
        self.real_samples = real_samples
        self.fake_samples = fake_samples

        all_samples = real_samples + fake_samples
        all_labels = [1] * len(real_samples) + [0] * len(fake_samples)
        samples = list(zip(all_samples, all_labels, strict=False))
        random.shuffle(samples)

        if continue_from is not None:
            df = continue_from
            processed_samples = set(df["path"].apply(Path))
            samples = [(sample, label) for sample, label in samples if sample not in processed_samples]
            write_mode = "a"
            logger.info("Continuing evaluation with {} remaining samples", len(samples))
        else:
            df = pd.DataFrame(columns=["path", "label", "pred", "response"])
            write_mode = "w"
            logger.info(
                "Starting evaluation of {} samples ({} real, {} fake)",
                len(samples),
                len(real_samples),
                len(fake_samples),
            )

        y_true = []
        y_pred = []

        pbar = tqdm(enumerate(samples), total=len(samples), desc="Eval...")
        for i, (sample, label) in pbar:
            logger.debug("Processing sample {}/{}: {}", i, len(samples), sample.name)
            pbar.set_description(f"Eval {sample.name[:19]}")

            try:
                response = self.llm_query(sample, label, should_print_response)
                pred = self.postprocess(response)
                if pred != -1:
                    y_true.append(label)
                    y_pred.append(pred)
                else:
                    logger.warning("Invalid prediction for sample: {}", sample.name)

                new_row = pd.DataFrame(
                    {
                        "path": [sample],
                        "label": [label],
                        "pred": [pred],
                        "response": [response],
                    }
                )

                if i == 0 and write_mode == "w":
                    new_row.to_csv(output_path, index=False, mode="w")
                else:
                    new_row.to_csv(output_path, index=False, mode="a", header=False)

                self._update_metrics(y_true, y_pred, pbar)

            except Exception as e:
                logger.error("Error processing sample {}: {}", sample.name, e)
                continue

        if not y_pred:
            logger.error("No valid predictions were made during evaluation")
            return 0.0, 0.0, 0.0

        return self._calculate_final_metrics(y_true, y_pred)

    @staticmethod
    def _calculate_final_metrics(y_true: list, y_pred: list) -> tuple[float, float, float]:
        """
        Calculate final evaluation metrics.

        Parameters
        ----------
        y_true : list
            List of true labels (1 for real, 0 for fake).
        y_pred : list
            List of predicted labels (1 for real, 0 for fake).

        Returns
        -------
        tuple[float, float, float]
            A tuple of (accuracy, precision, recall) metrics.
        """
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            real_accuracy = len([1 for i, y in enumerate(y_true) if y == 1 and y_pred[i] == 1]) / max(
                1, y_true.count(1)
            )
            fake_accuracy = len([1 for i, y in enumerate(y_true) if y == 0 and y_pred[i] == 0]) / max(
                1, y_true.count(0)
            )

            logger.info(
                "Metrics\n"
                "- Overall Accuracy: {:.3f} %\n"
                "- Precision: {:.3f} %\n"
                "- Recall: {:.3f} %\n"
                "- Acc. on real images: {:.3f} %\n"
                "- Acc. on fake images: {:.3f} %",
                accuracy * 100,
                precision * 100,
                recall * 100,
                real_accuracy * 100,
                fake_accuracy * 100,
            )
            return accuracy, precision, recall
        except Exception as e:
            logger.error("Error calculating metrics: {}", e)
            return 0.0, 0.0, 0.0
