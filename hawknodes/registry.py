"""Model lists, and the per-model widget groups behind each dropdown.

`models.json` seeds the dropdowns. After a successful chat call the LLM list also
merges whatever ``GET /v1/models`` reported, cached to `.models_cache.json` so it
survives a restart. Dropdowns are built once when ComfyUI starts, so a freshly
discovered model appears on the *next* launch -- `model_override` covers the gap.

Nothing in here may raise during import: a failed refresh must never stop the pack
from loading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from comfy_api.latest import IO

logger = logging.getLogger("HawkNodes")

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_FILE = os.path.join(_PACKAGE_ROOT, "models.json")
_CACHE_FILE = os.path.join(_PACKAGE_ROOT, ".models_cache.json")

CUSTOM_SLUG = "custom (use model_override)"
SIZE_DEFAULT = "default (model)"

# Model discovery returns bare ids with no capability flags, so vision support is
# guessed from the name. A false negative just means you use `custom` instead.
_VISION_HINTS = re.compile(
    r"(vl|vision|multimodal|omni|gpt-4o|gpt-4\.|gpt-5|gemini|claude|llava|pixtral|intern-?vl)",
    re.IGNORECASE,
)

_refresh_started = False


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("HawkNodes: could not read %s (%s)", path, exc)
        return {}


_MODELS = _load_json(_MODELS_FILE)


def _sizes(group: str) -> list[str]:
    sizes = _MODELS.get("sizes", {}).get(group)
    return list(sizes) if sizes else [SIZE_DEFAULT]


# ------------------------------------------------------------------- discovery


def cached_slugs() -> list[str]:
    cache = _load_json(_CACHE_FILE)
    slugs = cache.get("llm")
    return list(slugs) if isinstance(slugs, list) else []


def _write_cache(slugs: list[str]) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as handle:
            json.dump({"llm": sorted(set(slugs))}, handle, indent=2)
    except Exception as exc:
        logger.warning("HawkNodes: could not write the model cache (%s)", exc)


async def _refresh(api_url: str, api_key: str) -> None:
    from .client import fetch_models

    discovered = await fetch_models(api_url, api_key)
    if not discovered:
        return
    known = {spec["slug"] for spec in _MODELS.get("llm", [])} | set(cached_slugs())
    if not set(discovered) - known:
        return
    _write_cache(list(known | set(discovered)))
    logger.info(
        "HawkNodes: discovered %d Atlas models; restart ComfyUI to see them in the "
        "model dropdown.",
        len(discovered),
    )


def schedule_refresh(api_url: str, api_key: str) -> None:
    """Fire-and-forget model discovery, at most once per process."""
    global _refresh_started
    if _refresh_started:
        return
    _refresh_started = True
    try:
        asyncio.get_running_loop().create_task(_refresh(api_url, api_key))
    except Exception as exc:
        logger.debug("HawkNodes: could not schedule model discovery (%s)", exc)


# ----------------------------------------------------------------- LLM options


def _vision_inputs(max_images: int = 8) -> list:
    return [
        IO.Autogrow.Input(
            "images",
            template=IO.Autogrow.TemplateNames(
                IO.Image.Input("image"),
                names=[f"image_{i}" for i in range(1, max_images + 1)],
                min=0,
            ),
            tooltip=(
                "Optional image(s) for vision models, sent inline as base64. "
                "Downscaled to `image_max_side` first."
            ),
        ),
        IO.Int.Input(
            "image_max_side",
            default=1024,
            min=64,
            max=4096,
            step=64,
            advanced=True,
            tooltip=(
                "Longest edge of each image before encoding. A full 1024px PNG is "
                "roughly 1.5 MB of base64, so several large frames can exceed the "
                "request size limit."
            ),
        ),
        IO.Combo.Input(
            "image_format",
            options=["png", "jpeg"],
            default="png",
            advanced=True,
            tooltip="jpeg makes a much smaller request; png keeps sharp text legible.",
        ),
        IO.Combo.Input(
            "image_detail",
            options=["auto", "low", "high"],
            default="auto",
            advanced=True,
            tooltip="Vision detail hint. Ignored by models that do not support it.",
        ),
    ]


def llm_options() -> list:
    """DynamicCombo options for the LLM node.

    Vision-capable models expose image inputs; text-only ones do not, so images
    cannot be silently dropped on the floor.
    """
    options = [
        IO.DynamicCombo.Option(
            CUSTOM_SLUG,
            _vision_inputs()
            + [
                IO.String.Input(
                    "model_override",
                    default="",
                    placeholder="e.g. deepseek-ai/DeepSeek-V3-0324",
                    tooltip="Any Atlas model id. Required when `custom` is selected.",
                ),
            ],
        )
    ]

    seen = {CUSTOM_SLUG}
    for spec in _MODELS.get("llm", []):
        slug = spec.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        options.append(
            IO.DynamicCombo.Option(slug, _vision_inputs() if spec.get("vision") else [])
        )

    for slug in cached_slugs():
        if slug in seen:
            continue
        seen.add(slug)
        options.append(
            IO.DynamicCombo.Option(
                slug, _vision_inputs() if _VISION_HINTS.search(slug) else []
            )
        )

    return options


# --------------------------------------------------------------- image options


def _image_param_inputs(group: str, *, edit: bool) -> list:
    if group in ("gpt_image_2", "gpt_image_2_edit"):
        inputs = [
            IO.Combo.Input(
                "size",
                options=_sizes("gpt_image_2"),
                default="1024x1024",
                tooltip="Output resolution. These are gpt-image-2's official presets.",
            ),
            IO.Combo.Input(
                "quality",
                options=["low", "medium", "high"],
                default="medium",
                tooltip="Higher quality costs more and takes longer.",
            ),
        ]
        if edit:
            inputs.append(
                IO.Combo.Input(
                    "input_fidelity",
                    options=["high", "low"],
                    default="high",
                    tooltip=(
                        "How closely to preserve the reference images. high keeps "
                        "faces and logos intact; low allows more creative freedom."
                    ),
                )
            )
        return inputs

    return [
        IO.Combo.Input(
            "size",
            options=_sizes("generic"),
            default=SIZE_DEFAULT,
            tooltip=(
                "Leave on 'default (model)' to omit `size` entirely and let the model "
                "use its own native resolution -- safest for non-OpenAI models. Use "
                "`extra_params` for anything else this model accepts."
            ),
        ),
    ]


def image_options(kind: str) -> list:
    """DynamicCombo options for the t2i (``kind='t2i'``) or i2i (``'edit'``) node."""
    edit = kind == "edit"
    options = []
    seen = set()

    for spec in _MODELS.get(kind, []):
        slug = spec.get("slug")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        options.append(
            IO.DynamicCombo.Option(
                slug, _image_param_inputs(spec.get("params", "generic"), edit=edit)
            )
        )

    options.append(
        IO.DynamicCombo.Option(
            CUSTOM_SLUG,
            _image_param_inputs("generic", edit=edit)
            + [
                IO.String.Input(
                    "model_override",
                    default="",
                    placeholder="e.g. black-forest-labs/flux-schnell",
                    tooltip="Any Atlas image model id. Required when `custom` is selected.",
                ),
            ],
        )
    )
    return options


def resolve_slug(selection: dict) -> str:
    """Pull the model id out of a DynamicCombo value, honouring `model_override`."""
    slug = selection.get("model") or ""
    override = (selection.get("model_override") or "").strip()
    if override:
        return override
    if slug == CUSTOM_SLUG:
        raise ValueError(
            "Model is set to `custom` but `model_override` is empty. Type an Atlas "
            "model id into `model_override`, or pick a model from the dropdown."
        )
    return slug
