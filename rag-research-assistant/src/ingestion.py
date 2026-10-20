
from pathlib import Path
import pymupdf


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract native text or use OCR for scanned pages."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must be a PDF.")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):

            # First, try ordinary text extraction.
            text = page.get_text("text").strip()

            # If no text exists, use OCR.
            if not text:
                print(f"Running OCR on page {page_number}...")

                text_page = page.get_textpage_ocr(
                    language="eng",
                    dpi=200,
                    full=True
                )

                text = page.get_text(
                    "text",
                    textpage=text_page
                ).strip()

            # Preserve the original page number.
            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    if not pages:
        raise ValueError(
            "No text could be extracted, even with OCR. "
            "Check the PDF quality and OCR language."
        )

    return pages




def save_extracted_text(
    pages: list[dict],
    output_path: str
) -> None:
    """Save extracted text with page boundaries."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        for page in pages:
            file.write(f"\n--- Page {page['page_number']} ---\n")
            file.write(page["text"])
            file.write("\n")

    print(f"Extracted {len(pages)} pages.")
    print(f"Saved text to: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from a research paper PDF."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF")
    args = parser.parse_args()

    extracted_pages = extract_pdf(args.pdf_path)

    output_path = (
        Path("data/extracted")
        / f"{Path(args.pdf_path).stem}.txt"
    )

    save_extracted_text(extracted_pages, str(output_path))
