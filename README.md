# PDF Vision Parser

A PDF parsing tool that uses OpenAI's vision models to convert PDFs to accurate markdown format. Designed to handle complex layouts including tables, multi-row cells, and multi-page documents.

## Features

- **Vision-based parsing**: Uses Azure OpenAI GPT-4 Vision models for accurate content extraction
- **Complex layout support**: Handles tables, multi-row cells, hierarchical structures
- **Multi-page documents**: Context-aware parsing across pages
- **Smart table merging**: Automatically merges tables that span multiple pages
- **LLM post-processing**: Removes hallucinated content and fixes broken table rows
- **High accuracy**: Preserves numbers, dates, formatting, and document structure

## Use Cases

- Purchase orders and invoices
- Bill of materials
- Financial documents
- Technical specifications
- Any structured document with complex tables

## Architecture

```
PDF → Images (pdf2image) → Vision Model (page-by-page) → Raw Markdown
                                      ↓
                         LLM Post-Processing (optional)
                                      ↓
                    Clean, Merged Markdown Output
```

## Requirements

### Python Dependencies
See `requirements.txt`:
- `openai>=1.58.0` - Azure OpenAI client
- `pdf2image>=1.17.0` - PDF to image conversion
- `Pillow>=10.0.0` - Image processing
- `python-dotenv>=1.0.0` - Environment variable management

### System Dependencies

**Poppler** (required for pdf2image):

- **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases)
  - Extract to a local directory (e.g., `C:\Users\YourName\poppler-25.07.0`)
  - Note the path to the `bin` directory

- **Linux**:
  ```bash
  sudo apt-get install poppler-utils
  ```

- **macOS**:
  ```bash
  brew install poppler
  ```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Poppler** (see System Dependencies above)

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # For Azure OpenAI (if USE_AZURE=True)
   AZURE_API_KEY=your_azure_openai_api_key_here

   # For Standard OpenAI (if USE_AZURE=False)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Configuration

### Choosing Between Azure and Standard OpenAI

The parser supports both Azure OpenAI and standard OpenAI API. Choose which one to use in `vision_parser.py` (line 326):

```python
USE_AZURE = True  # Set to False to use standard OpenAI API
```

### API Settings

**For Azure OpenAI** (edit `get_openai_config()` function, lines 23-31):
```python
if use_azure:
    return {
        'use_azure': True,
        'api_key': os.getenv("AZURE_API_KEY"),
        'azure_endpoint': "https://your-resource.openai.azure.com/...",
        'api_version': "2024-12-01-preview",
        'model': 'gpt-4.1',  # Your deployment name
    }
```

**For Standard OpenAI** (edit `get_openai_config()` function, lines 33-37):
```python
else:
    return {
        'use_azure': False,
        'api_key': os.getenv("OPENAI_API_KEY"),
        'model': 'gpt-4o-2024-11-20',  # Model name
    }
```

### Other Settings

**Poppler path** (line 363):
```python
poppler_path=r"C:\Users\YourName\poppler-25.07.0\Library\bin"
```

**PDF path** (line 367):
```python
pdf_path = r"your_document.pdf"
```

## Usage

### Basic Usage

```python
from vision_parser import VisionParser, get_openai_config

# Choose API (Azure or Standard OpenAI)
openai_config = get_openai_config(use_azure=True)  # or False for standard OpenAI

# Initialize parser
parser = VisionParser(
    openai_config=openai_config,
    custom_prompt=custom_prompt,  # Optional
    poppler_path=r"C:\path\to\poppler\bin",
    use_context=True  # Enable context-aware parsing
)

# Convert PDF to markdown
markdown_pages = parser.convert_pdf(
    pdf_path="document.pdf",
    dpi=200,  # Image resolution
    clean_output=True  # Enable LLM post-processing
)

# Save to file
output_path = "output.md"
parser.save_markdown(markdown_pages, output_path)
```

### Command Line

Simply run:
```bash
python vision_parser.py
```

The script will:
1. Convert the PDF specified in the code
2. Parse each page with vision model
3. Apply LLM post-processing to clean and merge
4. Save the output as `{filename}_output.md`

### Advanced Options

**Disable post-processing** (get raw per-page output):
```python
markdown_pages = parser.convert_pdf(
    pdf_path="document.pdf",
    clean_output=False
)
```

**Disable context awareness** (parse each page independently):
```python
parser = VisionParser(
    openai_config=openai_config,
    use_context=False
)
```

**Adjust image quality**:
```python
markdown_pages = parser.convert_pdf(
    pdf_path="document.pdf",
    dpi=300  # Higher DPI = better quality but larger images
)
```

### Custom Prompts

Customize the parsing behavior:

```python
custom_prompt = """
Convert this document to markdown with focus on:
- Extracting all numerical data
- Preserving table structures
- Capturing specific fields: order numbers, dates, amounts
- Maintaining hierarchical relationships

Return ONLY markdown content.
"""

parser = VisionParser(
    openai_config=openai_config,
    custom_prompt=custom_prompt
)
```

## How It Works

### 1. PDF to Images
- Converts each PDF page to a high-resolution image (default 200 DPI)
- Uses poppler's `pdftoppm` under the hood

### 2. Vision Model Processing
- Sends each image to Azure OpenAI Vision API
- Provides context from previous page for continuity
- Extracts content as markdown with custom instructions

### 3. LLM Post-Processing (Optional)
- Removes hallucinated empty rows
- Merges tables split across pages
- Fixes broken table rows
- Ensures consistency and readability

