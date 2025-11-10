"""
This example demonstrates how to use the DoclingParser class with all available
configuration options.
"""

import os
import sys

# Add parent directory to path to import docling_parser module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.docling_parser import DoclingParser, parse_document


def main():
    # Create parser with custom configuration
    parser = DoclingParser(
        enable_ocr=True,                    # Enable OCR for scanned documents
        enable_table_structure=True,        # Extract table structures
        enable_formula_enrichment=True,     # Enrich formulas and equations
        num_threads=4,                      # Number of processing threads
        ocr_engine="rapidocr"          # OCR engine: "tesseract_cli", "tesseract", "easyocr", "rapidocr"
    )

    print(f"✓ Parser initialized")
    print(f"  - Docling available: {parser.docling_available}")
    print(f"  - Device: {parser.device if hasattr(parser, 'device') else 'N/A'}")
    print(f"  - Supported formats: {', '.join(parser.supported_extensions)}")
    
    pdf_path = "../doc.docx"  #input PDF file
    # Check if file is supported
    if parser.is_supported_file(pdf_path):
        print(f"  - File format {parser.get_file_extension(pdf_path)} is supported")

        # Parse document with markdown format
        result = parser.parse_document(
            file_path=pdf_path,
            use_markdown=True  # Convert to markdown
        )

        print(f"\n✓ Parsing complete with markdown format")
        print(f"  - Content length: {result['content_length']} characters")
        print(f"  - Word count: {result['word_count']} words")

        # Print parsed markdown to CLI
        print(f"\n{'='*80}")
        print("PARSED MARKDOWN CONTENT")
        print(f"{'='*80}\n")
        print(result['text_content'])
        print(f"\n{'='*80}")
        print("END OF CONTENT")
        print(f"{'='*80}\n")

        # Save to file in markdown format
        output_path = pdf_path.replace('.pdf', '_docling_output.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text_content'])
        print(f"✓ Saved to: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
