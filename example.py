"""
This example demonstrates how to use the VisionParser class with all available
configuration options set during initialization.
"""

import os
from vision_parser import VisionParser, get_openai_config


def main():

    #Get OpenAI configuration (Azure or Standard OpenAI)
    config = get_openai_config(use_azure=True)
    custom_prompt = "" #Optional

    # Create VisionParser
    print("\nCreating VisionParser...")
    parser = VisionParser(
        openai_config=config, # Required: OpenAI configuration
        custom_prompt=custom_prompt, # Optional: Custom parsing instructions
        poppler_path=None, # Optional: Path to poppler bin (auto-detected if None)
        use_context=True, # Whether to provide context from previous pages (improves multi-page documents)
        dpi=200, # Image resolution for PDF conversion (higher = better quality, slower)
        clean_output=True # Enable LLM post-processing to clean and merge tables across pages
    )
    print("✓ Parser initialized")

    # Parse PDF 
    pdf_path = "test1.pdf"  # Change to your PDF file

    if not os.path.exists(pdf_path):
        print(f"\n✗ Error: PDF file not found: {pdf_path}")
        return

    print(f"\nParsing PDF '{pdf_path}'...")
    print("Converting PDF to images...")
    print("Parsing each page with vision model...")

    # Convert PDF
    markdown_pages = parser.convert_pdf(pdf_path)

    print(f"✓ Parsing complete!")
    print(f"  - Pages processed: {len(markdown_pages)}")
    print(f"  - Output type: {'Cleaned and merged' if parser.clean_output else 'Raw per-page'}")

    # Preview the output
    print("\nPreview of parsed content...")
    if len(markdown_pages) > 1000:
        # Show first 500 characters
        preview = markdown_pages[0][:500]
        print(preview)
        print("\n... (preview truncated)")
    else:
        print("Parsed markdown:\n\n")
        print(markdown_pages[0])

    # Save to file
    output_filename = pdf_path.replace('.pdf', '_output.md')
    print(f"\nSaving output to '{output_filename}'...")
    parser.save_markdown(markdown_pages, output_filename)
    print(f"✓ Markdown saved successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
