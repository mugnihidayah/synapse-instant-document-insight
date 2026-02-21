"""
Document loaders for various formats

Supports PDF, DOCX, TXT files and images (PNG, JPG, JPEG, WEBP) via OCR
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import easyocr
import fitz  # PyMuPDF
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from PIL import Image

from src.core.exceptions import DocumentProcessingError

# mapping of file extensions to loader classes
LOADER_MAPPING: dict[str, type] = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

# Image extensions handled via OCR
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Global OCR reader (lazy-loaded)
_ocr_reader: easyocr.Reader | None = None


def _get_ocr_reader() -> easyocr.Reader:
    """Get cached easyocr reader instance."""
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en", "id"], gpu=False)
    return _ocr_reader


def get_supported_extensions() -> list[str]:
    """
    Get list of supported file extensions

    Returns:
      List of extension strings
    """
    return list(LOADER_MAPPING.keys()) + list(IMAGE_EXTENSIONS)


def _ocr_image(image: Image.Image) -> str:
    """
    Extract text from a PIL Image using easyocr.

    Args:
        image: PIL Image object

    Returns:
        Extracted text string
    """
    import numpy as np

    reader = _get_ocr_reader()
    # easyocr accepts numpy array
    img_array = np.array(image.convert("RGB"))
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


def _extract_pdf_image_text(pdf_path: str) -> dict[int, str]:
    """
    Extract text from embedded images in a PDF via OCR.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict mapping page index to OCR text from embedded images
    """
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

                    import io

                    pil_image = Image.open(io.BytesIO(image_bytes))

                    # Skip tiny images (icons, bullets, etc.)
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
        pass  # If image extraction fails, we still have the regular text

    return ocr_texts


def load_document_from_path(file_path: str | Path) -> list[Document]:
    """
    Load document from file path

    Args:
      file_path: Path to document file

    Returns:
      List of langchain documents

    Raises:
      DocumentProcessingError: If file extension is not supported or loading fails
    """

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


def load_document_from_upload(uploaded_file: Any, filename: str) -> list[Document]:
    """
    Load document from uploaded file

    Args:
      uploaded_file: file like object
      filename: original filename

    Returns:
      List of langchain documents

    Raises:
      DocumentProcessingError: If file extension is not supported or loading fails
    """

    ext = os.path.splitext(filename)[1].lower()

    all_supported = set(LOADER_MAPPING.keys()) | IMAGE_EXTENSIONS
    if ext not in all_supported:
        raise DocumentProcessingError(
            "Unsupported file format",
            details={"filename": filename, "supported": get_supported_extensions()},
        )

    # Handle image files via OCR
    if ext in IMAGE_EXTENSIONS:
        return _load_image_from_upload(uploaded_file, filename, ext)

    # Handle document files (PDF, DOCX, TXT)
    # save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader_class = LOADER_MAPPING[ext]
        loader = loader_class(tmp_path)
        documents: list[Document] = loader.load()

        # update metadata with original filename and document info
        total_pages = len(documents)
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["source_type"] = ext.lstrip(".")
            doc.metadata["total_pages"] = total_pages
            doc.metadata["content_origin"] = "text"
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1

        # For PDFs: extract text from embedded images via OCR
        if ext == ".pdf":
            ocr_texts = _extract_pdf_image_text(tmp_path)
            for page_idx, ocr_text in ocr_texts.items():
                if page_idx < len(documents):
                    documents[page_idx].page_content += f"\n\n[OCR from embedded image]\n{ocr_text}"
                    documents[page_idx].metadata["has_ocr"] = True
                    documents[page_idx].metadata["content_origin"] = "text+ocr"

        return documents
    except Exception as e:
        raise DocumentProcessingError(
            "Failed to load uploaded document", details={"filename": filename, "error": str(e)}
        ) from e

    finally:
        # clean up temp file
        os.unlink(tmp_path)


def _load_image_from_upload(uploaded_file: Any, filename: str, ext: str) -> list[Document]:
    """
    Load image file and extract text via OCR.

    Args:
        uploaded_file: file-like object
        filename: original filename
        ext: file extension (e.g. ".png")

    Returns:
        List with single Document containing OCR-extracted text
    """
    try:
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
    """
    Load multiple documents from uploaded files

    Args:
      uploaded_files: list of uploaded files

    Returns:
      Combined list of langchain documents
    """
    all_documents: list[Document] = []

    for uploaded_file in uploaded_files:
        docs = load_document_from_upload(uploaded_file, uploaded_file.name)
        all_documents.extend(docs)

    return all_documents
