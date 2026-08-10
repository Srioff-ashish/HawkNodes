"""HawkDocuments -- load PDF/DOC/DOCX/text files as LLM context."""

from __future__ import annotations

from comfy_api.latest import IO

from ..documents import DocumentPart, extract_document
from .common import (
    CATEGORY,
    HawkDocumentsType,
    document_options,
    resolve_document_path,
    upload_hint,
)


class HawkDocuments(IO.ComfyNode):
    """Chainable document loader.

    Parsing happens here rather than in the LLM node so ComfyUI caches the result:
    re-running a prompt against a 200 page PDF does not re-parse the PDF.
    """

    @classmethod
    def define_schema(cls):
        files = document_options(include_none=False)
        return IO.Schema(
            node_id="HawkDocuments",
            display_name="Hawk Documents",
            category=CATEGORY,
            description=(
                "Loads a PDF, Word document or text file as context for Hawk Atlas LLM. "
                "Chain several of these together to send more than one file in a "
                "single request."
            ),
            search_aliases=["pdf", "docx", "document", "atlas"],
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=files,
                    default=files[0] if files else None,
                    upload=IO.UploadType.model,
                    tooltip=upload_hint(),
                ),
                IO.String.Input(
                    "pages",
                    default="all",
                    tooltip="PDFs only. 'all', or a 1-based range such as '1-5' or '1,3,7-9'.",
                ),
                HawkDocumentsType.Input(
                    "documents",
                    optional=True,
                    tooltip="Optional documents from another Hawk Documents node, to batch them together.",
                ),
            ],
            outputs=[
                HawkDocumentsType.Output(
                    "documents", tooltip="Connect to Hawk Atlas LLM, or to another Hawk Documents node."
                ),
                IO.String.Output("text", tooltip="The extracted text, for previewing."),
            ],
        )

    @classmethod
    def execute(
        cls,
        file: str,
        pages: str = "all",
        documents: list[DocumentPart] | None = None,
    ) -> IO.NodeOutput:
        if not file:
            raise ValueError(
                "No file selected. Put a PDF, DOCX or text file in ComfyUI's input "
                "folder (or use the upload button) and reload the node."
            )

        part = extract_document(resolve_document_path(file), pages)
        combined = list(documents or []) + [part]
        preview = "\n\n".join(item.as_context() for item in combined)
        return IO.NodeOutput(combined, preview)
