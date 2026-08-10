# HawkNodes

Atlas Cloud nodes for ComfyUI. Send images, text, PDFs and Word documents to an LLM,
and generate or edit images — all through one Atlas API key.

Four nodes, under **Add Node → HawkNodes → Atlas**:

| Node | Does |
|---|---|
| **Hawk Atlas LLM** | images + text + PDF/DOCX/DOC/TXT/MD/CSV/JSON → text |
| **Hawk Documents** | loads a document as LLM context; chain several for multi-file input |
| **Hawk Atlas Text to Image** | prompt → image |
| **Hawk Atlas Image to Image** | reference image(s) + prompt → edited image |

Built on ComfyUI's V3 node API, so calls are `async` — a slow generation does not
freeze the queue, and **Cancel stops a running job** instead of waiting out the timeout.

**Every node previews its result in place.** The LLM node shows the reply, the image nodes
show what they generated, and the document loader shows the text it extracted — no need to
wire up a Preview node to see what happened. The outputs are still there for chaining.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Srioff-ashish/HawkNodes.git
pip install -r HawkNodes/requirements.txt
# optional, for scanned PDFs:  pip install pymupdf
# optional, for legacy .doc:   apt-get install antiword
```

Restart ComfyUI. Requires **ComfyUI 0.26.0 or newer** — older versions lack the node API
these nodes use, and will log a message saying so instead of loading.

`aiohttp`, `torch`, `numpy` and `pillow` already ship with ComfyUI, so `pypdf` and
`python-docx` are the only additions.

The same three commands work in a Colab ComfyUI setup — run them after ComfyUI is
installed and before you launch it. These nodes only call an API, so they need no GPU.

---

## The API key

Every node has an `api_key` widget under **Advanced**. Leave it **blank** and export the key
instead:

```bash
export ATLAS_API_KEY="your-key"      # before starting ComfyUI
```

> **Why blank is better.** ComfyUI writes every widget value into saved workflow JSON *and*
> into the metadata of PNGs it saves. A key typed into the widget travels with every
> workflow and every image you share. The environment variable never enters those files.

The widget wins when it is filled in, so either way works.

---

## Hawk Atlas LLM

Pick a model, write a prompt, optionally attach a document and images.

- **model** — a dropdown whose widgets change with the model. Vision-capable models show
  image inputs; text-only models do not. Choose **`custom (use model_override)`** to type
  any Atlas model id by hand — that option always includes image inputs.
- **document** — lists supported files in `ComfyUI/input/`, with an upload button. Set it to
  `(none)` if you are not sending a document.
- **context_chars** — the context-size budget for document text (default 120 000
  characters). Anything past it is cut, with a visible marker.
- **seed** — ComfyUI reuses a cached result when nothing changes, so **bump the seed to
  force a fresh call**. Set `control_after_generate` to `randomize` for a new call every run.

Outputs: `text`, `context_used` (exactly what document text was sent, plus any warnings —
check this first when a reply ignores your PDF), and `raw_json`.

### Documents

| Format | Handled by |
|---|---|
| `.pdf` | `pypdf`; the `pdf_pages` field takes `all`, `1-5` or `1,3,7-9` |
| `.docx` | `python-docx`, including table contents |
| `.doc` | `antiword` or LibreOffice if installed — otherwise re-save as `.docx` |
| `.txt` `.md` `.csv` `.json` | read directly |

A scanned PDF has no text layer. If `pymupdf` is installed those pages are rendered to
images and sent to the vision model instead; if it is not, `context_used` says so rather
than silently returning nothing.

To send several documents, chain **Hawk Documents** nodes into the LLM node's `documents`
input. Parsing happens in the loader, so re-running a prompt against a 200-page PDF does
not re-parse the PDF.

---

## The image nodes

Text-to-image and image-to-image both use Atlas's async generation endpoint: submit,
poll, download. A progress bar ticks while polling.

The **model dropdown changes the widgets below it**. `openai/gpt-image-2/*` shows `size`
and `quality` (and `input_fidelity` on the edit node); other models show a `size` that
defaults to **`default (model)`**, which omits `size` from the request entirely so the model
uses its own native resolution. Nothing you did not set is ever sent, because models
reject parameters they do not understand.

`extra_params` takes a JSON object merged into the request body, for anything a specific
model accepts that the node does not expose:

```json
{"guidance_scale": 3.5, "num_inference_steps": 28}
```

**Image to Image** takes one image plus a growing list of extra references — connect one
and another slot appears. `input_fidelity: high` preserves faces and logos; `low` allows
more creative freedom.

Note that `seed: 0` means "let the model choose", which also means ComfyUI will serve a
cached image on a re-run. Set `control_after_generate` to `randomize` for a different image
every time.

---

## Model lists

**Set `ATLAS_API_KEY` before starting ComfyUI and the dropdown fills itself in.** At startup
the pack calls `GET /v1/models` (~105 chat models at the time of writing) and caches the
result to `.models_cache.json`, refreshed daily. If discovery fails for any reason it logs
one line and falls back to the bundled list — it never delays or breaks startup.

Without the environment variable, the dropdown shows `models.json` plus anything cached from
a previous run. Discovery still runs after your first successful call, so those models appear
on the next restart.

Model ids whose name looks multimodal (`*-VL-*`, `*vision*`, `gemini*`, `claude*`, …) get
image inputs automatically.

> **Atlas model ids are case-sensitive and vendor-prefixed** — `deepseek-ai/DeepSeek-V3-0324`,
> `Qwen/Qwen3-Coder`, `zai-org/GLM-4.6`. The short names in Atlas's own documentation
> (`deepseek-v3`, `glm`, `minimax`) are marketing labels and are **rejected with a 400**.

`custom (use model_override)` reaches any model at any time. To make one permanent, add it to
`models.json`.

---

## Example graphs

```
Ask about a PDF and a picture
    LoadImage ──────────┐
                        ├──> Hawk Atlas LLM ──> Preview Text
    (document: spec.pdf)┘        (model: custom + a vision model id)

Expand a rough prompt, then generate
    Hawk Atlas LLM ──text──> Hawk Atlas Text to Image ──> Save Image
      "rewrite this as a detailed image prompt"

Edit with two references
    LoadImage ──image────┐
    LoadImage ──image_2──┴──> Hawk Atlas Image to Image ──> Save Image
                                (model: openai/gpt-image-2/edit)

Several documents at once
    Hawk Documents (a.pdf) ──> Hawk Documents (b.docx) ──documents──> Hawk Atlas LLM
```

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Nodes missing after restart | ComfyUI older than 0.26.0 — the console log says so |
| `No Atlas API key` | `api_key` blank *and* `ATLAS_API_KEY` unset |
| `Model is set to custom but model_override is empty` | type a model id into `model_override` |
| Reply ignores your document | read the `context_used` output; the document may have produced no text |
| Same result on every run | ComfyUI's cache — bump `seed` or set it to randomize |
| `400 ... "not found"` | the model id does not exist on your account. Ids are case-sensitive; `curl -s https://api.atlascloud.ai/v1/models -H "Authorization: Bearer $ATLAS_API_KEY"` lists valid ones |
| `Atlas API error 400` (image nodes) | the model rejected a parameter; try `size: default (model)`, or move it into `extra_params` |
| `.doc` fails | install `antiword`, or save the file as `.docx` |

---

## Testing without ComfyUI

```bash
python tests/smoke_test.py --offline                    # parsing, encoding, payloads
python tests/smoke_test.py --key $ATLAS_API_KEY         # + model list and a chat call
python tests/smoke_test.py --key $ATLAS_API_KEY --t2i --i2i
```

Every check reports independently, so one bad model id does not hide the rest.

---

 
MIT licensed.
