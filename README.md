# Document Parsers Library

A collection of document parsing tools that convert various document formats (PDF, DOCX, TXT) to markdown or structured text. Includes four specialized parsers:

1. **Vision Parser** - Uses OpenAI's vision models for accurate PDF parsing
2. **Docling Parser** - Uses Docling library for multi-format document parsing
3. **PyMuPDF Parser** - Fast local PDF parsing using PyMuPDF (no API required)
4. **DOCX Parser** - Fast local Word document parsing using python-docx

---

## Vision Parser

A vision-based PDF parser using OpenAI GPT-4 Vision models to extract content from complex documents.

### Key Features
- Vision-based parsing for complex layouts (tables, multi-row cells, hierarchical structures)
- Context-aware multi-page document processing
- Smart table merging across pages
- LLM post-processing to clean and merge content
- Supports both Azure OpenAI and standard OpenAI API

### Best For
Purchase orders, invoices, bill of materials, financial documents, technical specifications

### Requirements
- Python packages: `openai>=1.58.0`, `pdf2image`, `Pillow`, `python-dotenv`
- System: Poppler (auto-detected or manual installation)
- API Key: Azure OpenAI or OpenAI API key in `.env` file

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install Poppler
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
# Linux: sudo apt-get install poppler-utils
# macOS: brew install poppler

# Configure API keys in .env file
echo "AZURE_API_KEY=your_key" > .env
# OR
echo "OPENAI_API_KEY=your_key" > .env
```

### Usage

**Simple (Convenience Function):**
```python
from parsers.vision_parser import parse_pdf

markdown = parse_pdf(
    pdf_path="document.pdf",
    use_azure=True,
    output_path="output.md"
)
```

**Advanced (Full Control):**
```python
from parsers.vision_parser import VisionParser, get_openai_config

config = get_openai_config(use_azure=True)
parser = VisionParser(
    openai_config=config,
    custom_prompt=None,      # Optional custom instructions
    poppler_path=None,       # Auto-detected
    use_context=True,        # Context-aware parsing
    dpi=200,                 # Image quality
    clean_output=True        # LLM post-processing
)
markdown_pages = parser.convert_pdf("document.pdf")
parser.save_markdown(markdown_pages, "output.md")
```

**Run Example:**
```bash
python examples/vision_parser_example.py
```

---

## Docling Parser

A multi-format document parser using the Docling library for PDF, DOCX, and TXT files.

### Key Features
- Multi-format support (PDF, DOCX, TXT)
- Multiple OCR engines (Tesseract CLI, Tesseract, EasyOCR, RapidOCR)
- Table structure extraction with cell matching
- Formula/equation enrichment
- Automatic CUDA/CPU device detection
- No API costs (runs locally)

### Best For
Any document type including PDFs, Word documents, and text files where local processing is preferred

### Requirements
- Python packages: `docling`, `torch` (optional, for GPU), `psutil`
- No external API keys required
- Works offline

### Installation
```bash
# Install Docling
pip install docling

# Optional: Install PyTorch for GPU acceleration
pip install torch

# For memory management
pip install psutil
```

### Usage

**Simple (Convenience Function):**
```python
from parsers.docling_parser import parse_document

result = parse_document(
    file_path="document.pdf",
    enable_ocr=True,
    enable_table_structure=True,
    ocr_engine="tesseract_cli",  # Options: tesseract_cli, tesseract, easyocr, rapidocr
    output_path="output.md"
)
print(result['text_content'])
```

**Advanced (Full Control):**
```python
from parsers.docling_parser import DoclingParser

parser = DoclingParser(
    enable_ocr=True,                  # OCR for scanned docs
    enable_table_structure=True,      # Table extraction
    enable_formula_enrichment=True,   # Formula support
    num_threads=4,                    # Processing threads
    ocr_engine="tesseract_cli"        # OCR engine selection
)

result = parser.parse_document(
    file_path="document.pdf",
    use_markdown=True  # or False for structured text
)

# Access parsed content
text = result['text_content']
metadata = result['file_name'], result['word_count']
```

**Run Example:**
```bash
python examples/docling_example.py
```

---

## PyMuPDF Parser

A fast local PDF parser using PyMuPDF (fitz) library for text extraction without external APIs.

### Key Features
- Fast local PDF parsing (no API costs)
- Simple text extraction or structured output with positioning
- No OCR capabilities (fast but limited for scanned documents)
- No external dependencies beyond PyMuPDF
- Works offline

### Best For
Text-based PDFs where speed is important and no OCR is needed

### Requirements
- Python package: `PyMuPDF`
- No API keys required
- Works offline

### Installation
```bash
pip install PyMuPDF
```

### Usage

**Simple (Convenience Function):**
```python
from parsers.pymupdf_parser import parse_pdf

result = parse_pdf(
    file_path="document.pdf",
    use_markdown=True,  # Simple text
    output_path="output.txt"
)
print(result['text_content'])
```

**Advanced (Full Control):**
```python
from parsers.pymupdf_parser import PyMuPDFParser

parser = PyMuPDFParser()

result = parser.parse_document(
    file_path="document.pdf",
    use_markdown=False  # Structured with positioning
)

text = result['text_content']
metadata = result['file_name'], result['word_count']
```

**Run Example:**
```bash
python examples/pymupdf_example.py
```

---

## DOCX Parser

A fast local Word document parser using python-docx library for text extraction without external APIs.

### Key Features
- Fast local DOCX/DOC parsing (no API costs)
- Extract text from paragraphs and tables
- Simple text extraction or structured output with formatting
- Supports Word 2007+ documents
- Works offline

### Best For
Word documents where fast local extraction is sufficient

### Requirements
- Python package: `python-docx`
- No API keys required
- Works offline

### Installation
```bash
pip install python-docx
```

### Usage

**Simple (Convenience Function):**
```python
from parsers.docx_parser import parse_docx

result = parse_docx(
    file_path="document.docx",
    use_markdown=True,  # Simple text
    output_path="output.txt"
)
print(result['text_content'])
```

**Advanced (Full Control):**
```python
from parsers.docx_parser import DocxParser

parser = DocxParser()

result = parser.parse_document(
    file_path="document.docx",
    use_markdown=False  # Structured with formatting info
)

text = result['text_content']
metadata = result['file_name'], result['word_count']
```

**Run Example:**
```bash
python examples/docx_example.py
```

---


