"""Async HTTP layer for the Atlas Cloud API.

Two APIs live behind one key:

* chat completions -- OpenAI compatible, ``POST {base}/v1/chat/completions``
* image generation -- async, ``POST {base}/api/v1/model/generateImage`` returns a
  prediction id that is polled at ``GET {base}/api/v1/model/prediction/{id}``

Everything here is ``async`` so a slow Atlas call never blocks ComfyUI's event loop,
and the poll loop cooperates with the Cancel button.

This module deliberately has no ComfyUI imports at module level: ``tests/smoke_test.py``
exercises it standalone.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
from typing import Any

import aiohttp

logger = logging.getLogger("HawkNodes")

DEFAULT_CHAT_URL = "https://api.atlascloud.ai/v1"
DEFAULT_IMAGE_URL = "https://api.atlascloud.ai"

# Statuses worth another attempt. 4xx auth/validation errors are never retried --
# they will fail identically every time and just burn the user's wall clock.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

TERMINAL_OK = frozenset({"completed", "succeeded", "success"})
TERMINAL_FAIL = frozenset({"failed", "error", "canceled", "cancelled"})


class AtlasError(RuntimeError):
    """Any Atlas API failure. The message is shown on the node, so it carries the
    HTTP status and the response body -- that text is the whole diagnostic."""


def resolve_api_key(api_key: str | None) -> str:
    """Widget value wins; a blank widget falls back to the environment.

    Leaving the widget blank and exporting ``ATLAS_API_KEY`` is the safer setup:
    ComfyUI bakes widget values into saved workflows and PNG metadata, so a key
    typed into the node travels with every workflow you share.
    """
    key = (api_key or "").strip()
    if key:
        return key
    key = os.environ.get("ATLAS_API_KEY", "").strip()
    if key:
        return key
    raise AtlasError(
        "No Atlas API key. Either type one into the node's api_key widget, or set "
        "the ATLAS_API_KEY environment variable before starting ComfyUI."
    )


def normalize_chat_url(url: str | None) -> str:
    """Accept anything from ``api.atlascloud.ai`` to a full completions URL."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_CHAT_URL
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def normalize_image_base(url: str | None) -> str:
    """Strip any API suffix so we can append the image paths ourselves."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_IMAGE_URL
    for suffix in ("/api/v1", "/api", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url or DEFAULT_IMAGE_URL


def _interrupt_check():
    """ComfyUI's cooperative cancellation hook, or a no-op outside ComfyUI."""
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted

        return throw_exception_if_processing_interrupted
    except Exception:  # running under tests/smoke_test.py
        return lambda: None


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _read_json(response: aiohttp.ClientResponse) -> Any:
    text = await response.text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise AtlasError(
            f"Atlas returned non-JSON ({response.status}) from {response.url}: "
            f"{text[:500] or '(empty body)'}"
        ) from None


async def _request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    api_key: str,
    *,
    payload: dict | None = None,
    max_retries: int = 3,
) -> Any:
    """One request with exponential backoff on transient statuses."""
    last_error: str = ""
    for attempt in range(max_retries + 1):
        try:
            async with session.request(
                method, url, headers=_headers(api_key), json=payload
            ) as response:
                if response.status == 200:
                    return await _read_json(response)

                body = (await response.text())[:1000] or "(empty body)"
                if response.status not in RETRY_STATUSES or attempt == max_retries:
                    raise AtlasError(f"Atlas API error {response.status} at {url}: {body}")
                last_error = f"{response.status}: {body}"
        except aiohttp.ClientError as exc:
            if attempt == max_retries:
                raise AtlasError(f"Could not reach Atlas at {url}: {exc}") from exc
            last_error = str(exc)
        except asyncio.TimeoutError:
            if attempt == max_retries:
                raise AtlasError(
                    f"Atlas request to {url} timed out. Raise the node's `timeout` "
                    f"widget if the model is simply slow."
                ) from None
            last_error = "timeout"

        # Full jitter, so a burst of parallel nodes does not retry in lockstep.
        delay = min(2**attempt, 30) * (0.5 + random.random() / 2)
        logger.warning("HawkNodes: retrying %s in %.1fs (%s)", url, delay, last_error)
        await asyncio.sleep(delay)

    raise AtlasError(f"Atlas request to {url} failed: {last_error}")


# --------------------------------------------------------------------------- chat


async def chat_completion(
    api_url: str,
    api_key: str,
    payload: dict,
    *,
    timeout: int = 120,
    max_retries: int = 3,
) -> dict:
    """POST /v1/chat/completions and return the parsed response."""
    url = f"{normalize_chat_url(api_url)}/chat/completions"
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        result = await _request(
            session, "POST", url, api_key, payload=payload, max_retries=max_retries
        )
    if not isinstance(result, dict):
        raise AtlasError(f"Unexpected chat response shape: {type(result).__name__}")
    return result


def extract_message_text(response: dict) -> str:
    """Pull the assistant text out, turning every failure mode into a clear error."""
    error = response.get("error")
    if error:
        if isinstance(error, dict):
            message = error.get("message") or json.dumps(error)
            code = error.get("code", "unknown")
            raise AtlasError(f"Atlas error ({code}): {message}")
        raise AtlasError(f"Atlas error: {error}")

    choices = response.get("choices") or []
    if not choices:
        raise AtlasError(f"Atlas returned no choices. Full response: {json.dumps(response)[:500]}")

    message = choices[0].get("message") or {}
    if message.get("refusal"):
        raise AtlasError(f"The model refused to respond: {message['refusal']}")

    content = message.get("content")
    if isinstance(content, list):
        # Some models answer with content parts rather than a bare string.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        finish = choices[0].get("finish_reason", "unknown")
        raise AtlasError(
            f"Atlas returned an empty message (finish_reason={finish}). "
            f"If this is `length`, raise max_tokens."
        )
    return content


async def fetch_models(api_url: str, api_key: str, *, timeout: int = 20) -> list[str]:
    """GET /v1/models. Returns [] rather than raising -- model discovery is a
    convenience and must never stop a node from working."""
    url = f"{normalize_chat_url(api_url)}/models"
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            result = await _request(session, "GET", url, api_key, max_retries=0)
    except Exception as exc:
        logger.debug("HawkNodes: model discovery failed (%s)", exc)
        return []

    entries = result.get("data") if isinstance(result, dict) else None
    if not isinstance(entries, list):
        return []
    return sorted(
        {entry["id"] for entry in entries if isinstance(entry, dict) and entry.get("id")}
    )


# -------------------------------------------------------------------------- images


def _decode_output(item: str) -> bytes | str:
    """Outputs arrive as a data URI, an http URL, or bare base64. Return bytes for
    the first and third, and the URL string for the second (it needs a download)."""
    if item.startswith("data:"):
        return base64.b64decode(item.split(",", 1)[-1])
    if item.startswith("http://") or item.startswith("https://"):
        return item
    return base64.b64decode(item)


async def generate_image(
    api_url: str,
    api_key: str,
    payload: dict,
    *,
    poll_interval: float = 2.0,
    timeout: int = 300,
    max_retries: int = 3,
    progress=None,
) -> list[bytes]:
    """Submit a generation job, poll to completion, return raw image bytes.

    Text-to-image and image-to-image are the same endpoint: i2i just adds an
    ``images`` array of data URIs and an ``/edit`` model.
    """
    base = normalize_image_base(api_url)
    submit_url = f"{base}/api/v1/model/generateImage"
    check_interrupt = _interrupt_check()

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        submitted = await _request(
            session, "POST", submit_url, api_key, payload=payload, max_retries=max_retries
        )

        data = submitted.get("data") if isinstance(submitted, dict) else None
        prediction_id = data.get("id") if isinstance(data, dict) else None
        if not prediction_id:
            raise AtlasError(
                f"Atlas did not return a prediction id. Response: {json.dumps(submitted)[:500]}"
            )

        poll_url = f"{base}/api/v1/model/prediction/{prediction_id}"
        deadline = asyncio.get_event_loop().time() + timeout
        outputs: list[str] = []

        while True:
            # Checked before sleeping too, so Cancel lands within one interval.
            check_interrupt()

            if asyncio.get_event_loop().time() > deadline:
                raise AtlasError(
                    f"Generation {prediction_id} did not finish within {timeout}s. "
                    f"Raise the node's `timeout` widget for slower models."
                )

            result = await _request(session, "GET", poll_url, api_key, max_retries=1)
            data = result.get("data") if isinstance(result, dict) else {}
            status = str(data.get("status", "unknown")).lower()

            if status in TERMINAL_OK:
                outputs = data.get("outputs") or []
                if not outputs:
                    raise AtlasError(
                        f"Generation {prediction_id} completed but returned no images."
                    )
                break

            if status in TERMINAL_FAIL:
                raise AtlasError(
                    f"Generation failed: {data.get('error') or 'no reason given'}"
                )

            if progress is not None:
                progress.update(1)
            await asyncio.sleep(poll_interval)

        images: list[bytes] = []
        for item in outputs:
            decoded = _decode_output(item)
            if isinstance(decoded, bytes):
                images.append(decoded)
                continue
            async with session.get(decoded) as response:
                if response.status != 200:
                    raise AtlasError(
                        f"Could not download generated image ({response.status}): {decoded}"
                    )
                images.append(await response.read())

    return images
