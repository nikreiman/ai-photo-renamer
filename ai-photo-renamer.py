#!/usr/bin/env -S uv run --script
"""Analyze and rename images with AI using LM Studio."""

import argparse
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lmstudio
import yaml
from PIL import Image
from PIL.ExifTags import TAGS

log = logging.getLogger("main")


class ConfigFileNotFoundError(ValueError):
    """Raised when a config file cannot be found."""

    def __init__(self, config_path: str) -> None:
        """Initialize the exception."""
        super().__init__(f"{config_path} was not found")


class InvalidConfigFileError(ValueError):
    """Exception raised when a configuration file is invalid."""

    def __init__(self, config_path: str) -> None:
        """Initialize the exception."""
        super().__init__(f"{config_path} could not be parsed")


class PersonNotFoundError(Exception):
    """Raised when a person's image cannot be found."""

    def __init__(self, name: str, people_dir: str) -> None:
        """Initialize the exception."""
        super().__init__(f"Image for '{name}' not found in {people_dir}")


@dataclass
class Configuration:
    """Configuration file for a directory of images."""

    description: str | None
    people: list[str]
    dogs: list[str]


def configure_logging(*, verbose: bool) -> None:
    """Configure logging."""
    log_format = "%(levelname)s: %(message)s"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    muted_loggers = [
        "AsyncWebsocketHandler",
        "AsyncWebsocketThread",
        "ChatResponseEndpoint",
        "GetOrLoadEndpoint",
        "PIL.TiffImagePlugin",
        "RemoteCallHandler",
        "SyncLMStudioWebsocket",
        "asyncio",
        "httpcore.connection",
        "httpcore.http11",
        "httpx",
    ]
    for logger in muted_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)


def describe_images(
    image_dir: str,
    output_file: str,
    model: lmstudio.LLM,
    base_chat: lmstudio.Chat,
    description: str | None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Describe all images found in the given directory (and subdirectories)."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    # Images that haven't been renamed will just have a date as their filename, in the
    # form YYYY-MM-DD HH-MM-SS.
    renamed_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}.\d{2}.\d{2} - .*$")
    success_images = 0
    failed_images = 0
    skipped_images = 0

    for root, _, files in os.walk(image_dir):
        for file in sorted(files):
            file_path = Path(file)
            if file_path.suffix.lower() not in image_extensions:
                log.debug("Skipping non-image file %s", file_path)
                skipped_images += 1
                continue
            if renamed_pattern.match(file_path.stem):
                log.info("Skipping file %s, since it seems to have been renamed", file)
                skipped_images += 1
                continue

            chat = base_chat.copy()
            filepath = root / file_path
            log.info("Analyzing %s...", file)
            timestamp = get_image_timestamp(str(filepath))

            if dry_run:
                log.debug("Dry run, would prepare image")
            else:
                log.debug("Preparing image...")
                image_handle = lmstudio.prepare_image(filepath)

            user_message = (
                "Describe this image in 8 words, using the people you were told to "
                "identify. If you don't see any people you were told to identify, then "
                "disregard this instruction and just describe the general image. In this "
                "case you don't need to mention that you couldn't identify people."
            )
            if description:
                user_message += (
                    " It may also help you to know this about the image: "
                    f"{description}"
                )

            if not dry_run:
                chat.add_user_message(user_message, images=[image_handle])

            try:
                if dry_run:
                    log.info("Dry run: would send this message to LLM: %s", user_message)
                else:
                    log.debug("Describing image...")
                    start_time = time.monotonic()
                    response = model.respond(chat)
                    elapsed = time.monotonic() - start_time
                    image_description = response.content.strip()
                    log.info("Description: %s (%.0f seconds)", image_description, elapsed)
                    image_description = sanitize_filename(image_description)

                    new_filename = f"{timestamp} - {image_description}{file_path.suffix}"
                    new_filepath = root / Path(new_filename)

                    with Path(output_file).open("a+") as f:
                        f.write(f'mv -v "{filepath.name}" "{new_filepath.name}"\n')
                success_images += 1

            except lmstudio.LMStudioTimeoutError:
                log.warning("Timed out trying to analyze image %s", file)
                failed_images += 1
            except lmstudio.LMStudioServerError:
                log.exception("Failed analyzing image: %s", file)
                failed_images += 1

    return success_images, failed_images, skipped_images


def get_file_timestamp(filepath: str) -> str:
    """Get the timestamp for a given file on disk.

    This method is used as a backup in case the image itself has no metadata.
    """
    log.info("Getting file timestamp")
    stat = Path(filepath).stat()
    timestamp = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    return timestamp.strftime("%Y-%m-%d %H.%M.%S")


def get_image_timestamp(image_path: str) -> str:
    """Get a timestamp from the image metadata."""
    try:
        img = Image.open(image_path)
        exifdata = img.getexif()

        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag in {"DateTimeOriginal", "DateTime"}:
                log.debug("Found timestamp from image metadata")
                dt = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007
                return dt.strftime("%Y-%m-%d %H.%M.%S")

        return get_file_timestamp(image_path)
    except Exception:  # noqa: BLE001
        return get_file_timestamp(image_path)


def identify_person(
    people_dir: str,
    name: str,
    model: lmstudio.LLM,
    chat: lmstudio.Chat,
    *,
    is_dog: bool,
    dry_run: bool = False,
) -> None:
    """Tell the LLM to identify a given person based on their image."""
    log.info("Identifying %s: %s", "dog" if is_dog else "person", name)
    image_path = Path(people_dir) / f"{name}.jpg"
    if not image_path.exists():
        image_path = Path(people_dir) / f"{name}.png"
        if not image_path.exists():
            raise PersonNotFoundError(name, people_dir)

    image_handle = lmstudio.prepare_image(image_path)
    user_message_prefix = f"Here is an image of a person named {name}"
    if is_dog:
        user_message_prefix = f"Here is an image of a dog named {name}."
    user_message = (
        f"{user_message_prefix}. Simply acknowledge that you have identified them and "
        "are ready to recognize them in future image prompts. Keep your response as "
        "brief as possible."
    )
    chat.add_user_message(user_message, images=[image_handle])

    if dry_run:
        log.info("Dry run: would send this message to LLM: %s", user_message)
    else:
        start_time = time.monotonic()
        response = model.respond(chat)
        elapsed = time.monotonic() - start_time
        log.info("LLM says: %s (%0.f seconds)", response.content, elapsed)


def main(args: argparse.Namespace) -> None:
    """Run the application's main loop."""
    configure_logging(verbose=args.verbose)

    lmstudio.configure_default_client(f"{args.lm_studio_ip}:{args.lm_studio_port}")
    lmstudio.set_sync_api_timeout(float(args.timeout))
    model_config = {"contextLength": args.context_length}
    log.info("Connecting to server and loading model %s", args.model)
    model = lmstudio.llm(args.model, config=model_config)  # noqa[arg-type]

    chat = lmstudio.Chat()
    chat.add_system_prompt(args.prompt)
    configuration = parse_configuration(args.image_dir)

    for person in configuration.people:
        identify_person(
            args.people_dir, person, model, chat, is_dog=False, dry_run=args.dry_run
        )
    for dog in configuration.dogs:
        identify_person(
            args.people_dir, dog, model, chat, is_dog=True, dry_run=args.dry_run
        )

    with Path(args.output).open("w") as f:
        f.write("#!/bin/sh\n")
    Path(args.output).chmod(0o755)
    log.info("Generated script: %s", args.output)

    start_time = time.monotonic()
    success_images, failed_images, skipped_files = describe_images(
        args.image_dir,
        args.output,
        model,
        chat,
        configuration.description,
        dry_run=args.dry_run,
    )
    elapsed = int(time.monotonic() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    log.info(
        "Analyzed %d files (%d failed, %d skipped), %s",
        success_images,
        failed_images,
        skipped_files,
        " ".join(parts),
    )


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and rename images using LM Studio",
    )

    parser.add_argument(
        "-c",
        "--context-length",
        default=20000,
        help="Context length to use for model",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Simulate what actions would be taken without actually running them",
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        help="Path to directory of images to analyze",
        required=True,
    )
    parser.add_argument(
        "--lm-studio-ip",
        default="10.0.0.4",
        help="IP address of LM Studio server",
    )
    parser.add_argument(
        "--lm-studio-port",
        default=1234,
        help="Port number of LM Studio server",
        type=int,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="qwen/qwen3-vl-4b",
        help="Model name to load",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Path to output shell script (will use `rename.sh` under --image-dir by "
            "default)"
        ),
    )
    parser.add_argument(
        "-p",
        "--people-dir",
        help="Path to directory that contains people images",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "You are a helpful AI that is good at describing images. For each image that "
            "you analyze, you are to describe it briefly. If you recognize any of the "
            "people you were told to identify, use their names in your description. If "
            "you don't see any people, then just describe the image itself. If you see "
            "other people but you don't recognize them, then don't say anything about "
            "them. Also, do not mention anything about orientation, namely things "
            "being upside-down."
        ),
        help="Default system prompt to use",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        default=300,
        help="Timeout to use when waiting for a response from LM Studio",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()
    if not args.output:
        args.output = Path(args.image_dir) / "rename.sh"

    return args


def parse_configuration(image_dir: str) -> Configuration:
    """Parse a configuration file from the image directory."""
    config_path = Path(image_dir) / ".analyze.yml"
    if not config_path.exists():
        raise ConfigFileNotFoundError(str(config_path))

    with config_path.open() as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise InvalidConfigFileError(str(config_path))

    return Configuration(
        description=data.get("description"),
        people=data.get("people", []),
        dogs=data.get("dogs", []),
    )


def sanitize_filename(filename: str) -> str:
    """Remove any evil characters from the description to make a suitable filename."""
    filename = re.sub(r'[<>:"/\\|?*\.]', "", filename.strip())
    filename = re.sub(r"\s+", " ", filename)
    return filename[:200]


if __name__ == "__main__":
    main(parse_args())
