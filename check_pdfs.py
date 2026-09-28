from pathlib import Path
from pypdf import PdfReader


DATA_DIR = Path("data/documents")


def check_pdfs():
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data/ folder.")
        return

    print("=" * 60)
    print("PDF TEXT READABILITY CHECK")
    print("=" * 60)

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)

            total_characters = 0

            for page in reader.pages:
                text = page.extract_text() or ""
                total_characters += len(text)

            print(f"\nFile: {pdf_file.name}")
            print(f"Pages: {len(reader.pages)}")
            print(f"Characters: {total_characters}")

            if total_characters == 0:
                print("Status: ❌ No text extracted")
            else:
                print("Status: ✅ Text readable")

        except Exception as e:
            print(f"\nFile: {pdf_file.name}")
            print("Status: ❌ Error reading PDF")
            print(f"Error: {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_pdfs()