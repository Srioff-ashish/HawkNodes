"""HawkAtlasLLM -- images, text and documents in, text out."""

from __future__ import annotations

import json
import logging

from comfy_api.latest import IO, ui

from .. import registry
from ..client import (
    DEFAULT_CHAT_URL,
    AtlasError,
    chat_completion,
    extract_message_text,
    normalize_chat_url,
    resolve_api_key,
)
from ..documents import DocumentPart, assemble_context, extract_document
from ..images import tensor_batch_to_data_uris
from .common import (
    CATEGORY,
    NO_DOCUMENT,
    HawkDocumentsType,
    connection_inputs,
    document_options,
    resolve_document_path,
    upload_hint,
)

logger = logging.getLogger("HawkNodes")


def _content_parts(text: str, image_uris: list[str], detail: str) -> list[dict]:
    parts: list[dict] = [{"type": "text", "text": text}]
    for uri in image_uris:
        image_url: dict = {"url": uri}
        if detail and detail != "auto":
            image_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": image_url})
    return parts


def _build_messages(
    system_prompt: str, text: str, image_uris: list[str], detail: str
) -> list[dict]:
    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    # A plain string when there is nothing to attach: some models reject the
    # content-parts form for text-only requests.
    content = _content_parts(text, image_uris, detail) if image_uris else text
    messages.append({"role": "user", "content": content})
    return messages


class HawkAtlasLLM(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        files = document_options()
        return IO.Schema(
            node_id="HawkAtlasLLM",
            display_name="Hawk Atlas LLM",
            category=CATEGORY,
            description=(
                "Calls an Atlas Cloud LLM with any mix of text, images and documents "
                "(PDF, DOCX, DOC, TXT, MD, CSV, JSON) and returns the reply as text."
            ),
            search_aliases=["atlas", "llm", "chat", "vision", "pdf"],
            inputs=[
                IO.String.Input(
                    "user_prompt",
                    multiline=True,
                    default="",
                    tooltip="The question or instruction for the model.",
                ),
                IO.DynamicCombo.Input(
                    "model",
                    options=registry.llm_options(),
                    tooltip=(
                        "Atlas model to call. Vision-capable models show image inputs; "
                        "pick `custom` to type any model id by hand."
                    ),
                ),
                IO.Combo.Input(
                    "document",
                    options=files,
                    default=NO_DOCUMENT,
                    upload=IO.UploadType.model,
                    tooltip=upload_hint(),
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "ComfyUI reuses a cached result when nothing changes, so bump "
                        "this (or set control_after_generate to randomize) to force a "
                        "fresh call. Also sent to Atlas as a sampling seed when above 0."
                    ),
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="You are a helpful assistant.",
                    optional=True,
                    advanced=True,
                    tooltip="Standing instructions that shape how the model responds.",
                ),
                HawkDocumentsType.Input(
                    "documents",
                    optional=True,
                    tooltip="Documents from one or more Hawk Documents nodes.",
                ),
                IO.String.Input(
                    "context_text",
                    default="",
                    optional=True,
                    force_input=True,
                    advanced=True,
                    tooltip="Extra context from another node, prepended to the documents.",
                ),
                IO.String.Input(
                    "pdf_pages",
                    default="all",
                    optional=True,
                    advanced=True,
                    tooltip="Pages to read from the `document` PDF: 'all', '1-5', or '1,3,7-9'.",
                ),
                IO.Int.Input(
                    "context_chars",
                    default=120000,
                    min=0,
                    max=4000000,
                    step=1000,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Context size: the character budget for document text. Anything "
                        "beyond it is cut, with a marker. 0 means no limit -- which can "
                        "overflow the model's context window."
                    ),
                ),
                IO.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    advanced=True,
                    tooltip="Higher is more varied, lower is more deterministic.",
                ),
                IO.Float.Input(
                    "top_p",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    advanced=True,
                    tooltip="Nucleus sampling. Usually left at 1.0 when using temperature.",
                ),
                IO.Int.Input(
                    "max_tokens",
                    default=2048,
                    min=1,
                    max=200000,
                    step=64,
                    optional=True,
                    advanced=True,
                    tooltip="Cap on the length of the reply, not the input.",
                ),
                IO.Boolean.Input(
                    "json_mode",
                    default=False,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "Ask for a JSON object back. Your prompt still has to describe "
                        "the shape you want. Not every model supports this."
                    ),
                ),
                IO.String.Input(
                    "stop",
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip="Comma-separated stop sequences. Leave blank for none.",
                ),
                IO.Float.Input(
                    "frequency_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    optional=True,
                    advanced=True,
                    tooltip="Above 0 discourages repeating the same tokens.",
                ),
                IO.Float.Input(
                    "presence_penalty",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    optional=True,
                    advanced=True,
                    tooltip="Above 0 encourages introducing new topics.",
                ),
                *connection_inputs(DEFAULT_CHAT_URL),
                IO.Int.Input(
                    "timeout",
                    default=180,
                    min=10,
                    max=3600,
                    optional=True,
                    advanced=True,
                    tooltip="Seconds to wait for the whole request.",
                ),
                IO.Int.Input(
                    "max_retries",
                    default=3,
                    min=0,
                    max=10,
                    optional=True,
                    advanced=True,
                    tooltip="Retries for rate limits and server errors. Auth errors are never retried.",
                ),
                IO.String.Input(
                    "extra_body_json",
                    multiline=True,
                    default="",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        'Extra JSON merged into the request body, e.g. {"top_k": 40}. '
                        "An escape hatch for parameters this node does not expose."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output("text", tooltip="The model's reply."),
                IO.String.Output(
                    "context_used",
                    tooltip="Exactly what document text was sent, plus any warnings.",
                ),
                IO.String.Output("raw_json", tooltip="The full Atlas response, for debugging."),
            ],
        )

    @classmethod
    async def execute(
        cls,
        user_prompt: str,
        model: dict,
        document: str = NO_DOCUMENT,
        seed: int = 0,
        system_prompt: str = "",
        documents: list[DocumentPart] | None = None,
        context_text: str = "",
        pdf_pages: str = "all",
        context_chars: int = 120000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        stop: str = "",
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        api_url: str = DEFAULT_CHAT_URL,
        api_key: str = "",
        timeout: int = 180,
        max_retries: int = 3,
        extra_body_json: str = "",
    ) -> IO.NodeOutput:
        slug = registry.resolve_slug(model)
        key = resolve_api_key(api_key)

        # ---- gather documents -------------------------------------------------
        parts: list[DocumentPart] = list(documents or [])
        if document and document != NO_DOCUMENT:
            parts.append(extract_document(resolve_document_path(document), pdf_pages))

        context, warnings = assemble_context(parts, context_text, context_chars)

        # ---- gather images ----------------------------------------------------
        supports_vision = "images" in model
        image_uris: list[str] = []
        if supports_vision:
            max_side = int(model.get("image_max_side", 1024))
            image_format = model.get("image_format", "png")
            for tensor in (model.get("images") or {}).values():
                if tensor is not None:
                    image_uris.extend(
                        tensor_batch_to_data_uris(
                            tensor, max_side=max_side, image_format=image_format
                        )
                    )
            for part in parts:
                image_uris.extend(part.images)
        else:
            scanned = sum(len(part.images) for part in parts)
            if scanned:
                warnings.append(
                    f"{scanned} scanned page(s) could not be sent: {slug} has no image "
                    f"input. Switch to a vision-capable model to read them."
                )

        if not user_prompt.strip() and not context and not image_uris:
            raise ValueError(
                "Nothing to send. Fill in user_prompt, select a document, or connect an image."
            )

        # ---- build the request ------------------------------------------------
        prompt_text = user_prompt.strip()
        if context:
            prompt_text = (
                f"{prompt_text}\n\n---\nDOCUMENT CONTEXT:\n{context}"
                if prompt_text
                else f"DOCUMENT CONTEXT:\n{context}"
            )

        payload: dict = {
            "model": slug,
            "messages": _build_messages(
                system_prompt, prompt_text, image_uris, model.get("image_detail", "auto")
            ),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed > 0:
            payload["seed"] = seed
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stop.strip():
            payload["stop"] = [s.strip() for s in stop.split(",") if s.strip()]
        if frequency_penalty:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty:
            payload["presence_penalty"] = presence_penalty

        if extra_body_json.strip():
            try:
                extra = json.loads(extra_body_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"extra_body_json is not valid JSON: {exc}") from None
            if not isinstance(extra, dict):
                raise ValueError("extra_body_json must be a JSON object, e.g. {\"top_k\": 40}")
            payload.update(extra)

        # ---- call -------------------------------------------------------------
        try:
            response = await chat_completion(
                api_url, key, payload, timeout=timeout, max_retries=max_retries
            )
        except AtlasError as exc:
            # Atlas answers an unknown model with a bare 400 "not found", which
            # says nothing about which field was wrong. Name the model instead.
            message = str(exc)
            if "400" in message and "not found" in message.lower():
                raise AtlasError(
                    f"Atlas rejected the model id {slug!r} (400 not found). That model "
                    f"is not available on your account.\n\n"
                    f"List the ids you can actually use:\n"
                    f"    curl -s {normalize_chat_url(api_url)}/models "
                    f"-H \"Authorization: Bearer $ATLAS_API_KEY\"\n\n"
                    f"Then set `model` to `{registry.CUSTOM_SLUG}` and put a working id in "
                    f"`model_override`, or add it to models.json.\n\n"
                    f"Original error: {message}"
                ) from None
            raise

        text = extract_message_text(response)

        # Cheap once-per-session model discovery, now that we know the key works.
        registry.schedule_refresh(api_url, key)

        context_report = context
        if warnings:
            notes = "\n".join(f"- {warning}" for warning in warnings)
            context_report = f"[HawkNodes notes]\n{notes}\n\n{context}".rstrip()
            for warning in warnings:
                logger.warning("HawkNodes: %s", warning)

        # Show the reply on the node, so reading it does not require wiring up a
        # preview node. Warnings go first -- they explain a disappointing answer.
        preview = f"⚠ {' | '.join(warnings)}\n\n{text}" if warnings else text

        return IO.NodeOutput(
            text,
            context_report,
            json.dumps(response, indent=2, ensure_ascii=False),
            ui=ui.PreviewText(preview),
        )


__all__ = ["HawkAtlasLLM"]
