"""
Document loaders for various formats.

Supports PDF, DOCX, TXT files and images (PNG, JPG, JPEG, WEBP) via OCR.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from PIL import Image

from src.core.exceptions import DocumentProcessingError

LOADER_MAPPING: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_ocr_engine = None


def _get_ocr_engine():
    """Get cached RapidOCR engine instance."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
    return _ocr_engine


def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions."""
    return list(LOADER_MAPPING.keys()) + list(IMAGE_EXTENSIONS)


def _ocr_image(image: Image.Image) -> str:
    """Extract text from a PIL Image using RapidOCR."""
    import numpy as np

    engine = _get_ocr_engine()
    img_array = np.array(image.convert("RGB"))
    result, _ = engine(img_array)
    if not result:
        return ""
    return "\n".join(item[1] for item in result)


def _extract_pdf_image_text(pdf_path: str) -> dict[int, str]:
    """Extract OCR text from embedded images in a PDF."""
    ocr_texts: dict[int, str] = {}

    try:
        pdf_doc = fitz.open(pdf_path)
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            images = page.get_images(full=True)
            if not images:
                continue

            page_ocr_parts: list[str] = []
            for img_info in images:
                xref = img_info[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    if pil_image.width < 50 or pil_image.height < 50:
                        continue

                    text = _ocr_image(pil_image)
                    if text.strip():
                        page_ocr_parts.append(text.strip())
                except Exception:
                    continue

            if page_ocr_parts:
                ocr_texts[page_idx] = "\n".join(page_ocr_parts)

        pdf_doc.close()
    except Exception:
        pass

    return ocr_texts


def _extract_pdf_table_text(pdf_path: str) -> dict[int, str]:
    """Extract basic table text from PDF pages if table detector is available."""
    table_texts: dict[int, str] = {}

    try:
        pdf_doc = fitz.open(pdf_path)
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            if not hasattr(page, "find_tables"):
                continue

            try:
                tables = page.find_tables()
            except Exception:
                continue

            if not tables or not tables.tables:
                continue

            page_tables: list[str] = []
            for table in tables.tables:
                rows = table.extract()
                row_lines = [" | ".join((cell or "").strip() for cell in row) for row in rows]
                serialized = "\n".join(row_lines).strip()
                if serialized:
                    page_tables.append(serialized)

            if page_tables:
                table_texts[page_idx] = "\n\n".join(page_tables)

        pdf_doc.close()
    except Exception:
        pass

    return table_texts


def load_document_from_path(file_path: str | Path) -> list[Document]:
    """Load document from file path."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise DocumentProcessingError("File not found", details={"path": str(file_path)})

    ext = file_path.suffix.lower()
    if ext not in LOADER_MAPPING:
        raise DocumentProcessingError(
            "Unsupported file format",
            details={"extension": ext, "supported_extensions": get_supported_extensions()},
        )

    loader_class = LOADER_MAPPING[ext]
    loader = loader_class(str(file_path))

    try:
        documents: list[Document] = loader.load()
    except Exception as e:
        raise DocumentProcessingError(
            "Failed to load document", details={"path": str(file_path), "error": str(e)}
        ) from e

    return documents


def load_document_from_upload(
    uploaded_file: Any,
    filename: str,
    *,
    enable_ocr: bool = True,
    extract_tables: bool = False,
) -> list[Document]:
    """
    Load document from uploaded file.

    Args:
      uploaded_file: File-like object
      filename: Original filename
      enable_ocr: Enable OCR for image/PDF image content
      extract_tables: Enable lightweight PDF table extraction

    Returns:
      List of LangChain documents
    """
    ext = os.path.splitext(filename)[1].lower()

    all_supported = set(LOADER_MAPPING.keys()) | IMAGE_EXTENSIONS
    if ext not in all_supported:
        raise DocumentProcessingError(
            "Unsupported file format",
            details={"filename": filename, "supported": get_supported_extensions()},
        )

    if ext in IMAGE_EXTENSIONS:
        return _load_image_from_upload(uploaded_file, filename, ext, enable_ocr=enable_ocr)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader_class = LOADER_MAPPING[ext]
        loader = loader_class(tmp_path)
        documents: list[Document] = loader.load()

        total_pages = len(documents)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["source_type"] = ext.lstrip(".")
            doc.metadata["total_pages"] = total_pages
            doc.metadata["content_origin"] = "text"
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1

        if ext == ".pdf" and enable_ocr:
            ocr_texts = _extract_pdf_image_text(tmp_path)
            for page_idx, ocr_text in ocr_texts.items():
                if page_idx < len(documents):
                    documents[page_idx].page_content += f"\n\n[OCR from embedded image]\n{ocr_text}"
                    documents[page_idx].metadata["has_ocr"] = True
                    documents[page_idx].metadata["content_origin"] = "text+ocr"

        if ext == ".pdf" and extract_tables:
            table_texts = _extract_pdf_table_text(tmp_path)
            for page_idx, table_text in table_texts.items():
                if page_idx < len(documents):
                    documents[page_idx].page_content += f"\n\n[Extracted table]\n{table_text}"
                    documents[page_idx].metadata["has_tables"] = True
                    existing = documents[page_idx].metadata.get("content_origin", "text")
                    if "table" not in existing:
                        documents[page_idx].metadata["content_origin"] = f"{existing}+table"

        return documents

    except Exception as e:
        raise DocumentProcessingError(
            "Failed to load uploaded document",
            details={"filename": filename, "error": str(e)},
        ) from e

    finally:
        os.unlink(tmp_path)


def _load_image_from_upload(
    uploaded_file: Any,
    filename: str,
    ext: str,
    *,
    enable_ocr: bool = True,
) -> list[Document]:
    """Load image file and extract text via OCR."""
    if not enable_ocr:
        raise DocumentProcessingError(
            "OCR is disabled for image uploads",
            details={"filename": filename, "extension": ext},
        )

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        pil_image = Image.open(uploaded_file)
        text = _ocr_image(pil_image)

        if not text.strip():
            raise DocumentProcessingError(
                "No text could be extracted from image",
                details={"filename": filename},
            )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "source_type": "image",
                    "total_pages": 1,
                    "content_origin": "ocr",
                },
            )
        ]
    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            "Failed to process image", details={"filename": filename, "error": str(e)}
        ) from e


def load_documents_from_uploads(uploaded_files: list) -> list[Document]:
    """Load multiple documents from uploaded files."""
    all_documents: list[Document] = []

    for uploaded_file in uploaded_files:
        docs = load_document_from_upload(uploaded_file, uploaded_file.name)
        all_documents.extend(docs)

    return all_documents
