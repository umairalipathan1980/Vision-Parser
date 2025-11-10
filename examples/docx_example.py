"""
This example demonstrates how to use the DocxParser class for Word document parsing.
"""

import os
import sys

# Add parent directory to path to import docx_parser module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsers.docx_parser import DocxParser, parse_docx


def main():
    print("\n" + "="*80)
    print("DOCX WORD DOCUMENT PARSER")
    print("="*80 + "\n")

    # Step 1: Create parser
    print("Step 1: Creating DocxParser...")
    parser = DocxParser()

    print(f"✓ Parser initialized")
    print(f"  - Supported formats: {', '.join(parser.supported_extensions)}")

    # Step 2: Parse DOCX file
    docx_path = "../doc.docx"  # Input DOCX file

    if not os.path.exists(docx_path):
        print(f"\n✗ Error: DOCX file not found: {docx_path}")
        print("Please update the docx_path variable with a valid DOCX file.")
        print("\nAlternatively, you can use the convenience function:")
        print("  result = parse_docx('your_document.docx')")
        return

    # Check if file is supported
    if parser.is_supported_file(docx_path):
        print(f"  - File format {os.path.splitext(docx_path)[1]} is supported")

        print(f"\nStep 2: Parsing DOCX '{docx_path}'...")

        # Parse document with markdown format (simple text)
        result = parser.parse_document(
            file_path=docx_path,
            use_markdown=True  # Simple text extraction
        )

        print(f"✓ Parsing complete!")
        print(f"  - Content length: {result['content_length']} characters")
        print(f"  - Word count: {result['word_count']} words")
        print(f"  - Parsing method: {result['parsing_method']}")
        print(f"  - Format used: {result['format_used']}")

        # Step 3: Print parsed content to CLI
        print(f"\n{'='*80}")
        print("PARSED CONTENT")
        print(f"{'='*80}\n")
        print(result['text_content'])
        print(f"\n{'='*80}")
        print("END OF CONTENT")
        print(f"{'='*80}\n")

        # Step 4: Save to markdown file
        output_path = docx_path.replace('.docx', '_docx_output.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text_content'])
        print(f"✓ Saved to: {output_path}")

        # Step 5 (Optional): Parse with structured format
        print(f"\n{'='*80}")
        print("BONUS: Parsing with structured format...")
        print(f"{'='*80}\n")

        result_structured = parser.parse_document(
            file_path=docx_path,
            use_markdown=False  # Structured extraction with formatting info
        )

        print(f"✓ Structured parsing complete!")
        print(f"  - Content length: {result_structured['content_length']} characters")
        print(f"  - Format used: {result_structured['format_used']}")

        # Preview structured output
        preview = result_structured['text_content'][:500]
        print(f"\nPreview of structured output:\n{preview}\n... (truncated)")

        # Save structured output as markdown
        output_structured = docx_path.replace('.docx', '_docx_structured.md')
        with open(output_structured, 'w', encoding='utf-8') as f:
            f.write(result_structured['text_content'])
        print(f"✓ Structured output saved to: {output_structured}")

    print("\n" + "="*80)
    print("Example complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
