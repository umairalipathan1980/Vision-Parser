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

### Poppler Path

The parser will automatically detect poppler installation in common locations:
- **Windows**: User home directory, C:/, Program Files
- **Linux/macOS**: System PATH

You can also specify a custom path if needed:
```python
poppler_path=r"C:\Users\YourName\poppler-25.07.0\Library\bin"
```

## Usage

The `vision_parser` module can be used in three ways:

### Method 1: Simple - Convenience Function (Recommended)

The easiest way to get started:

```python
from vision_parser import parse_pdf

# Parse a PDF with one function call (poppler path auto-detected)
markdown = parse_pdf(
    pdf_path="document.pdf",
    use_azure=True,
    output_path="output.md",
    print_output=True  # Print to console
)

# Access the parsed markdown
print(markdown[0])
```

**Note:** The `poppler_path` parameter is optional and will be auto-detected. Specify it only if you have a custom installation location.

### Method 2: Advanced - Using the VisionParser Class

For more control over the parsing process. **All options are set during initialization:**

```python
from vision_parser import VisionParser, get_openai_config

# Step 1: Get configuration
config = get_openai_config(use_azure=True)  # or False for OpenAI

# Step 2: (Optional) Custom parsing instructions
custom_prompt = """
Extract data focusing on:
- Tables and numerical data
- Preserve exact formatting
Return only markdown.
"""

# Step 3: Initialize parser with ALL options
parser = VisionParser(
    openai_config=config,
    custom_prompt=custom_prompt,              # Optional
    poppler_path=None,                         # Optional - auto-detected if None
    use_context=True,                          # Context-aware multi-page parsing
    dpi=200,                                   # Image resolution
    clean_output=True                          # Enable LLM post-processing
)

# Step 4: Parse PDF - only needs file path!
markdown_pages = parser.convert_pdf("document.pdf")

# Step 5: Save output
parser.save_markdown(markdown_pages, "output.md")
```

### Method 3: Run the Example

A comprehensive example is provided in `example.py`:

```bash
python example.py
```

The example demonstrates all available configuration options including custom prompts, DPI settings, context awareness, and LLM post-processing

### Customization Options

All options are configured when creating the `VisionParser` instance:

**Disable post-processing** (get raw per-page output):
```python
parser = VisionParser(
    openai_config=config,
    clean_output=False  # Disable LLM post-processing
)
markdown_pages = parser.convert_pdf("document.pdf")
```

**Disable context awareness** (parse each page independently):
```python
parser = VisionParser(
    openai_config=config,
    use_context=False  # Each page parsed independently
)
```

**Adjust image quality**:
```python
parser = VisionParser(
    openai_config=config,
    dpi=300  # Higher DPI = better quality but slower
)
```

**Specify custom poppler path** (if auto-detection fails):
```python
parser = VisionParser(
    openai_config=config,
    poppler_path=r"C:\custom\path\to\poppler\bin"
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
    openai_config=config,
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

