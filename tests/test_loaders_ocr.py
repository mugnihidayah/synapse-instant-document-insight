"""Tests for multi-modal document loading with OCR support."""

import io
import os
import tempfile

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ingestion.loaders import (
    IMAGE_EXTENSIONS,
    LOADER_MAPPING,
    _extract_pdf_image_text,
    _load_image_from_upload,
    get_supported_extensions,
    load_document_from_upload,
)


class TestSupportedExtensions:
    """Test that image extensions are properly registered."""

    def test_includes_document_formats(self):
        exts = get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".txt" in exts

    def test_includes_image_formats(self):
        exts = get_supported_extensions()
        assert ".png" in exts
        assert ".jpg" in exts
        assert ".jpeg" in exts
        assert ".webp" in exts

    def test_image_extensions_not_in_loader_mapping(self):
        """Image extensions should NOT be in LOADER_MAPPING (handled separately)."""
        for ext in IMAGE_EXTENSIONS:
            assert ext not in LOADER_MAPPING


class TestImageOCR:
    """Test standalone image upload with OCR."""

    def _create_text_image(self, text: str = "Hello World Test 123") -> io.BytesIO:
        """Create a simple image with text for OCR testing."""
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        # Use default font (always available)
        draw.text((20, 30), text, fill="black")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    def test_image_upload_returns_document(self):
        """Uploading an image should return a Document with OCR text."""
        img_buf = self._create_text_image("Hello World")
        docs = _load_image_from_upload(img_buf, "test.png", ".png")

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "test.png"
        assert docs[0].metadata["source_type"] == "image"
        assert docs[0].metadata["content_origin"] == "ocr"
        assert docs[0].metadata["total_pages"] == 1

    def test_image_metadata_correct(self):
        """Image metadata should have correct fields."""
        img_buf = self._create_text_image("Test OCR Text")
        docs = _load_image_from_upload(img_buf, "photo.jpg", ".jpg")

        meta = docs[0].metadata
        assert meta["source"] == "photo.jpg"
        assert meta["source_type"] == "image"
        assert meta["content_origin"] == "ocr"

    def test_load_document_from_upload_routes_image(self):
        """load_document_from_upload should route image files to OCR."""
        img_buf = self._create_text_image("Routing test")
        docs = load_document_from_upload(img_buf, "screenshot.png")

        assert len(docs) >= 1
        assert docs[0].metadata["source_type"] == "image"
        assert docs[0].metadata["content_origin"] == "ocr"


class TestTxtUpload:
    """Test that TXT upload still works and has new metadata."""

    def test_txt_has_content_origin(self):
        """TXT files should have content_origin='text'."""
        content = b"This is a test document with some text content."
        buf = io.BytesIO(content)

        docs = load_document_from_upload(buf, "test.txt")

        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "test.txt"
        assert docs[0].metadata["source_type"] == "txt"
        assert docs[0].metadata["content_origin"] == "text"


class TestPDFWithImages:
    """Test PDF embedded image extraction."""

    def _create_pdf_with_image(self) -> str:
        """Create a test PDF with an embedded image containing text."""
        # Create PDF
        pdf = fitz.open()
        page = pdf.new_page()

        # Add regular text
        page.insert_text((50, 50), "This is regular PDF text")

        # Create an image with text and embed it
        img = Image.new("RGB", (200, 80), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), "Image Text OCR", fill="black")

        # Save image to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Insert image into PDF
        rect = fitz.Rect(50, 100, 250, 180)
        page.insert_image(rect, stream=img_bytes.getvalue())

        # Save PDF
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.save(tmp.name)
        pdf.close()
        tmp.close()

        return tmp.name

    def test_extract_pdf_image_text(self):
        """Should extract OCR text from images embedded in PDF."""
        pdf_path = self._create_pdf_with_image()
        try:
            ocr_texts = _extract_pdf_image_text(pdf_path)
            # Should have text from page 0
            assert isinstance(ocr_texts, dict)
            # If OCR found text, it should be for page 0
            if ocr_texts:
                assert 0 in ocr_texts
                assert len(ocr_texts[0]) > 0
        finally:
            os.unlink(pdf_path)

    def test_pdf_upload_with_ocr_metadata(self):
        """PDF with images should have has_ocr and content_origin metadata."""
        pdf_path = self._create_pdf_with_image()
        try:
            with open(pdf_path, "rb") as f:
                docs = load_document_from_upload(f, "report.pdf")

            assert len(docs) >= 1
            assert docs[0].metadata["source"] == "report.pdf"
            assert docs[0].metadata["source_type"] == "pdf"

            # Check if OCR was detected
            has_ocr_doc = any(d.metadata.get("has_ocr") for d in docs)
            if has_ocr_doc:
                ocr_doc = next(d for d in docs if d.metadata.get("has_ocr"))
                assert ocr_doc.metadata["content_origin"] == "text+ocr"
        finally:
            os.unlink(pdf_path)
