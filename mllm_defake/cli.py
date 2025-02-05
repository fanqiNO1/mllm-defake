import os
import random
import re
import sys
from pathlib import Path

import click
import orjson
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mllm_defake
from mllm_defake.classifiers.mllm_classifier import MLLMClassifier
from mllm_defake.vllms import VLLM

SUPPORTED_MODELS = ["gpt4o", "gpt4omini", "llama32vi", "llavacot", "qvq"]
SUPPORTED_DATASETS = ["WildFakeResampled", "ImageFolders", ""]


def find_prompt_file(prompt: str) -> dict:
    if Path(prompt).exists():
        p = Path(prompt)
        logger.info("Loading prompt file: {}", p)
        return orjson.loads(p.read_bytes())
    prompts_dir = Path("prompts")
    for path in prompts_dir.rglob("*.json"):
        if path.stem == prompt:
            logger.info("Assuming prompt file: {}", path)
            return orjson.loads(path.read_bytes())
    raise FileNotFoundError(f"Prompt file not found: {prompt}")


def load_model(model: str) -> VLLM:
    model = model.lower()
    if model == "gpt-4o" or model == "gpt4o":
        from mllm_defake.vllms import GPT4o

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set but required for GPT-4o."
            )
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy is None:
            logger.warning("Starting GPT-4o inference without a proxy.")
        return GPT4o(api_key=api_key, proxy=proxy)
    elif model == "gpt-4o-mini" or model == "gpt4omini":
        from mllm_defake.vllms import GPT4oMini

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set but required for GPT-4o-mini."
            )
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy is None:
            logger.warning("Starting GPT-4o-mini inference without a proxy.")
        return GPT4oMini(api_key=api_key, proxy=proxy)
    elif model == "llama-3.2-vision-instruct" or model == "llama32vi":
        from mllm_defake.vllms import Llama32VisionInstruct

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return Llama32VisionInstruct(api_key=api_key, base_url=base_url)
    elif model == "llama-3.2-vision-cot" or model == "llavacot":
        from mllm_defake.vllms import Llama32VisionCoT

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return Llama32VisionCoT(api_key=api_key, base_url=base_url)
    elif model == "qvq":
        from mllm_defake.vllms import QVQ

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return QVQ(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Invalid model: {model}")


def load_samples(
    dataset: str, count: int, real_dir: Path | None = None, fake_dir: Path | None = None
) -> tuple[list[Path], list[Path]]:
    if dataset == "WildFakeResampled":
        from mllm_defake.defake_dataset import WildFakeResampled

        dataset = WildFakeResampled()
        real_samples, fake_samples = dataset.list_images()
        if count < len(real_samples):
            real_samples = random.sample(real_samples, count)
            logger.debug(
                f"Sampling {count} / {len(real_samples)} real samples from {dataset}"
            )
        if count < len(fake_samples):
            fake_samples = random.sample(fake_samples, count)
            logger.debug(
                f"Sampling {count} / {len(fake_samples)} fake samples from {dataset}"
            )
        logger.info(
            "Loaded {} real samples and {} fake samples from {}",
            len(real_samples),
            len(fake_samples),
            dataset,
        )
        return real_samples, fake_samples
    elif dataset == "ImageFolders":
        from mllm_defake.defake_dataset import ImageFolders

        if not real_dir or not fake_dir:
            raise ValueError(
                "If `--dataset` is empty, `--real_dir` and `--fake_dir` must be specified."
            )
        dataset = ImageFolders(real_dir, fake_dir)
        real_samples, fake_samples = dataset.list_images()
        if count < len(real_samples):
            real_samples = random.sample(real_samples, count)
            logger.debug(
                f"Sampling {count} / {len(real_samples)} real samples from {dataset}"
            )
        if count < len(fake_samples):
            fake_samples = random.sample(fake_samples, count)
            logger.debug(
                f"Sampling {count} / {len(fake_samples)} fake samples from {dataset}"
            )
        logger.info(
            "Loaded {} real samples and {} fake samples from {}",
            len(real_samples),
            len(fake_samples),
            dataset,
        )
        return real_samples, fake_samples
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


@click.command()
@click.option(
    "-p",
    "--prompt",
    help="The name or path to the prompt v3 JSON file. Please refer to `prompts/readme.md` for format details. If a name is given, the corresponding JSON file should be placed under `prompts/` or its subdirectories. Note that the name of the prompt file must be unique, otherwise the first match will be used.",
    default="prompts/v3/simple.json",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    help="The model to use for inference.",
    default="llama32vi",
)
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(SUPPORTED_DATASETS),
    help="The dataset to use for inference. If not specified, will read `--real_dir` and `--fake_dir`.",
    default="",
)
@click.option(
    "--real_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="The directory containing real images. Must be specified if `--dataset` is not provided.",
    default="",
)
@click.option(
    "--fake_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="The directory containing fake images. Must be specified if `--dataset` is not provided.",
    default="",
)
@click.option(
    "-c",
    "--count",
    type=int,
    help="Will evaluate on `count` real samples AND `count` fake samples. If the dataset has fewer samples than the count, all samples will be used. Otherwise, the samples will be randomly selected. Note that the number of real and fake samples will be equal, summing up to `2 * count` samples. Defaults to 10.",
    default=10,
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="The random seed to use for sample selection. Defaults to 3706.",
    default=3706,
)
@click.option(
    "-l",
    "--log_id",
    help="The human-readable ID of the log file to be generated. The log file will be saved under `logs/`.",
    default="",
)
@click.option(
    "-o",
    "--output",
    help="The name or path to the output CSV file. If a name is given, the CSV file will be saved under `outputs/`. If not specified, a sensible default name will be generated.",
    default="",
)
@click.option(
    "--continue",
    "continue_eval",
    is_flag=True,
    help="Continue evaluation from a previous run if the output file exists.",
)
def infer(
    prompt,
    model,
    dataset,
    count,
    log_id,
    output,
    continue_eval,
    real_dir,
    fake_dir,
    seed,
):
    """
    This script evaluates the performance of a multimodal language model (MLLM) classifier on a dataset of real and fake images.

    The script uses a prompt file in JSON format to generate queries for the MLLM model. The model generates responses based on the queries, which are then used to classify the images as real or fake.

    It outputs a CSV file with the results of the classification.
    """
    random.seed(seed)
    # Setup paths to enable external decorators
    pwd = os.getcwd()
    if pwd not in sys.path:
        sys.path.append(pwd)

    # Create necessary directories
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = f"logs/{log_id}.log" if log_id else "logs/main.log"
    logger.add(log_file, rotation="2 MB", backtrace=True, diagnose=True)
    logger.debug(
        "Starting MLLM classifier with prompt={}, model={}, dataset={}, count={}, log_id={}, output={}, continue={}",
        prompt,
        model,
        dataset,
        count,
        log_id,
        output,
        continue_eval,
    )

    # Load prompt
    try:
        prompt = find_prompt_file(prompt)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        logger.error("Invalid JSON found for prompt `{}`: {}", prompt, e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading prompt `{}`: {}", prompt, e)
        sys.exit(1)

    # Determine output path
    if not output:
        readable_output_name = "{dataset}-{count}_{model}_{prompt}{log_id}.csv".format(
            dataset=dataset,
            count=count,
            model=model.short_name if hasattr(model, "short_name") else model,
            prompt=prompt["name"] if isinstance(prompt, dict) else Path(prompt).stem,
            log_id=f"_{log_id}" if log_id else "",
        )
        output_path = Path("outputs") / readable_output_name
    else:
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
        if not output_path.is_absolute():
            output_path = Path("outputs") / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Will save the output to {}", output_path.resolve())

    # Load model
    model = load_model(model)

    # Check for existing results if continuing
    continue_df = None
    if dataset == "" or dataset == "ImageFolders":
        if not real_dir or not fake_dir:
            raise ValueError(
                "If `--dataset` is empty, `--real_dir` and `--fake_dir` must be specified."
            )
        dataset = "ImageFolders"
        real_samples, fake_samples = load_samples(dataset, count, real_dir, fake_dir)
    else:
        real_samples, fake_samples = load_samples(dataset, count)
    if continue_eval:
        if output_path.exists():
            continue_df = pd.read_csv(output_path)
            already_processed_reals: int = continue_df["label"].sum()
            already_processed_fakes: int = len(continue_df) - already_processed_reals
            processed_reals: list[Path] = (
                continue_df[continue_df["label"] == 1]["path"].apply(Path).tolist()
            )
            processed_fakes: list[Path] = (
                continue_df[continue_df["label"] == 0]["path"].apply(Path).tolist()
            )
            logger.info(
                "Continuing evaluation from {} ({} real and {} fake samples already processed)",
                output_path,
                already_processed_reals,
                already_processed_fakes,
            )
            reduce_count_reals = max(0, len(real_samples) - already_processed_reals)
            reduce_count_fakes = max(0, len(fake_samples) - already_processed_fakes)
            real_samples = [s for s in real_samples if s not in processed_reals][
                :reduce_count_reals
            ]
            fake_samples = [s for s in fake_samples if s not in processed_fakes][
                :reduce_count_fakes
            ]
            logger.info(
                "Will evaluate on {} real and {} fake samples",
                len(real_samples),
                len(fake_samples),
            )
            logger.info("Continuing evaluation from {}", output_path)
        else:
            logger.warning(
                "Output file not found for continuing evaluation, restarting from scratch."
            )
    else:
        logger.info("Starting evaluation from scratch.")

    classifier = MLLMClassifier(prompt, model, real_samples, fake_samples)
    if not output:
        readable_output_name = "{dataset}-{count}_{model}_{prompt}{log_id}.csv".format(
            dataset=dataset,
            count=count,
            model=model.short_name if hasattr(model, "short_name") else model,
            prompt=prompt["name"],
            log_id=f"_{log_id} " if log_id else "",
        )
        readable_output_path: Path = Path("outputs") / readable_output_name
    else:
        if output.endswith(".csv"):
            readable_output_path = Path(output)
        else:
            readable_output_name = f"{output}.csv"
            readable_output_path = Path("outputs") / readable_output_name
    readable_output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting evaluation with output: {}", readable_output_path)
    classifier.evaluate(
        output_path=f"outputs/{readable_output_name}", continue_from=continue_df
    )


def guess_experiment_setup_from_path(path: Path) -> tuple[str, str, str, int]:
    name = path.stem

    # Check if the name is in the format of `dataset-count_model_prompt`
    # 1. match dataset from SUPPORTED_DATASETS
    def match_and_remove(full_str, sub_str) -> str:
        if full_str.startswith(sub_str + "-") or full_str.startswith(sub_str + "_"):
            return full_str[len(sub_str) + 1 :]
        return full_str

    dataset = ""
    for ds in SUPPORTED_DATASETS:
        new_name = match_and_remove(name, ds)
        if new_name != name:
            dataset = ds
            name = new_name
            break
    else:
        logger.warning("Could not determine dataset from the name: {}", name)
    # 2. match count
    count = 0
    count_str = re.search(r"^(\d+)_", name)
    if count_str:
        count = int(count_str.group(1))
        name = name[count_str.end() :]
    else:
        logger.warning("Could not determine count from the name: {}", name)
    # 3. match model from SUPPORTED_MODELS
    model = ""
    supported_models_sorted = sorted(SUPPORTED_MODELS, key=len, reverse=True)
    for m in supported_models_sorted:
        new_name = match_and_remove(name, m)
        if new_name != name:
            model = m
            name = new_name
            break
    else:
        logger.warning("Could not determine model from the name: {}", name)
    # 4. prompt is the remaining part
    prompt = name
    return dataset, model, prompt, count


def doc_writer(experiment_name: str) -> None:
    if Path(experiment_name).exists():
        df = pd.read_csv(experiment_name)
        real_experiment_name = Path(experiment_name).stem
    elif Path(f"outputs/{experiment_name}.csv").exists():
        df = pd.read_csv(f"outputs/{experiment_name}.csv")
        real_experiment_name = experiment_name
    else:
        raise FileNotFoundError(f"Experiment file not found for {experiment_name}")
    # Filter out failed predictions
    count_before = len(df)
    df = df[df["pred"] != -1]
    count_after = len(df)
    fails = count_before - count_after
    if count_before != count_after:
        logger.warning(
            "Filtered out {} failed predictions from the experiment",
            fails,
        )
    accuracy = accuracy_score(df["label"], df["pred"])
    precision = precision_score(df["label"], df["pred"], zero_division=0)
    recall = recall_score(df["label"], df["pred"], zero_division=0)
    real_accuracy = len(df[(df["label"] == 1) & (df["pred"] == 1)]) / max(
        1, df["label"].sum()
    )
    fake_accuracy = len(df[(df["label"] == 0) & (df["pred"] == 0)]) / max(
        1, (1 - df["label"]).sum()
    )

    metric_str = (
        "Metrics {}\n"
        + " - Overall Accuracy: {:.3f} %\n"
        + " - Precision: {:.3f} %\n"
        + " - Recall: {:.3f} %\n"
        + " - Acc. on real images: {:.3f} %\n"
        + " - Acc. on fake images: {:.3f} %\n"
    ).format(
        real_experiment_name,
        accuracy * 100,
        precision * 100,
        recall * 100,
        real_accuracy * 100,
        fake_accuracy * 100,
    )
    total_images = len(df)
    count_real = df["label"].sum()
    count_fake = total_images - count_real
    logger.info(metric_str)
    with open(f"outputs/{real_experiment_name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Experiment Results - `{real_experiment_name}`\n\n")
        f.write("## Metrics\n\n")
        f.write(
            f"- *{total_images} images, {count_real} real, {count_fake} generated.*\n\n"
        )
        if fails > 0:
            f.write(f"- *{fails} failed predictions were filtered out.*\n\n")
        f.write(f"```\n{metric_str}\n```\n\n")
        f.write("## Results\n\n")
        f.write("| Image | Label | Prediction | Full Response |\n")
        f.write("| --- | --- | --- | --- |\n")
        markdown_dir = Path("outputs/").resolve()
        for i, row in df.iterrows():
            image_name = Path(row["path"]).name
            image_path = Path(row["path"]).resolve()
            common_root = os.path.commonpath([image_path, markdown_dir])
            layers_up_to_common_root = len(markdown_dir.relative_to(common_root).parts)
            rel_to_image_path = image_path.relative_to(common_root)
            rel_path_str = "../" * layers_up_to_common_root + str(rel_to_image_path)
            response_edited = (
                row["response"]
                .replace("\n", "<br>")
                .replace(" ", " ")
                .replace("|", "\\|")
            )
            gt = "gen'd" if row["label"] == 0 else "real"
            pred = "gen'd" if row["pred"] == 0 else "real"
            f.write(
                f"| ![Image {image_name}]({rel_path_str}) | {gt} | {pred} | {response_edited} |\n"
            )
        f.write("\n")
        f.write(f"Created by {mllm_defake.__name__} - `v{mllm_defake.__version__}`")
    logger.success(
        "Saved the experiment results to outputs/{}.md", real_experiment_name
    )
    d_, m_, p_, c_ = guess_experiment_setup_from_path(Path(real_experiment_name))
    return {
        "experiment_name": real_experiment_name,
        "dataset": d_,
        "model": m_,
        "prompt": p_,
        "count": c_,
        "output_path": f"outputs/{real_experiment_name}.md",
        "total_images": total_images,
        "count_real": count_real,
        "count_fake": count_fake,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "fails": fails,
    }


@click.command()
@click.argument("experiment_name", type=str, default="all")
@click.option(
    "-s",
    "--summarize_to",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Summarize the results to a CSV file.",
    default=None,
)
def doc(experiment_name, summarize_to):
    """
    Prints the metrics for a given experiment, and then creates a markdown file that shows the image, label, prediction and model response in a tabular format.

    EXPERIMENT_NAME: The name of the experiment to compute the metrics for. You may use a CSV path instead of an experiment name. By default, the script will look for the CSV file under `outputs/`.
    """
    if experiment_name == "all":
        res = {}
        for path in Path("outputs/").rglob("*.csv"):
            res[path.stem] = doc_writer(path)
        if summarize_to:
            df = pd.DataFrame.from_dict(res, orient="index")
            df.to_csv(summarize_to)
            logger.success("Summarized the results to {}", summarize_to)
        return
    if summarize_to:
        logger.warning(
            "Summarizing to a CSV file is not supported for documenting singular experiment. Ignoring the `summarize_to` option."
        )
    doc_writer(experiment_name)


@click.group()
def main_cli():
    pass


main_cli.add_command(infer)
main_cli.add_command(doc)


if __name__ == "__main__":
    main_cli()
