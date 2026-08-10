"""Pieces shared by more than one node."""

from __future__ import annotations

import logging
import os

from comfy_api.latest import IO

from ..documents import SUPPORTED_EXTENSIONS, is_supported

logger = logging.getLogger("HawkNodes")

CATEGORY = "HawkNodes/Atlas"

#: Chained document payload -- a ``list[DocumentPart]`` passed between nodes.
HawkDocumentsType = IO.Custom("HAWK_DOCUMENTS")

NO_DOCUMENT = "(none)"


def input_directory() -> str:
    try:
        import folder_paths

        return folder_paths.get_input_directory()
    except Exception:  # pragma: no cover - only outside ComfyUI
        return ""


def document_options(include_none: bool = True) -> list[str]:
    """Supported files sitting in ComfyUI's input directory.

    Listed at schema build time, which is how the file dropdown and its Upload
    button work for every other loader node in ComfyUI.
    """
    directory = input_directory()
    files: list[str] = []
    if directory and os.path.isdir(directory):
        try:
            files = sorted(
                entry.name
                for entry in os.scandir(directory)
                if entry.is_file() and is_supported(entry.name)
            )
        except OSError as exc:
            logger.warning("HawkNodes: could not list %s (%s)", directory, exc)

    return ([NO_DOCUMENT] + files) if include_none else files


def resolve_document_path(filename: str) -> str:
    """Turn a dropdown value into a real path, honouring ComfyUI's ``[input]``
    style annotations that the upload widget produces."""
    try:
        import folder_paths

        return folder_paths.get_annotated_filepath(filename)
    except Exception:
        return os.path.join(input_directory(), filename)


def upload_hint() -> str:
    return (
        f"Pick a file from ComfyUI's input folder, or use the upload button. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def connection_inputs(default_url: str) -> list:
    """The api_url / api_key pair, identical on every node."""
    return [
        IO.String.Input(
            "api_url",
            default=default_url,
            advanced=True,
            tooltip="Atlas API base URL. Only change this for a proxy or a private deployment.",
        ),
        IO.String.Input(
            "api_key",
            default="",
            advanced=True,
            placeholder="blank = use the ATLAS_API_KEY environment variable",
            tooltip=(
                "Atlas API key. WARNING: ComfyUI saves widget values into workflow "
                "JSON and PNG metadata, so a key typed here travels with any workflow "
                "you share. Leaving it blank and exporting ATLAS_API_KEY instead keeps "
                "the key out of your files."
            ),
        ),
    ]


def progress_bar(total: int):
    """ComfyUI's progress bar, or a no-op stand-in outside ComfyUI."""
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except Exception:  # pragma: no cover - only outside ComfyUI

        class _Noop:
            def update(self, _value):
                pass

        return _Noop()
