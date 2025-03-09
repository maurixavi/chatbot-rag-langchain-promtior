from pathlib import Path

# Default URLs to fetch web-based documents
DEFAULT_URLS = [
    "https://www.promtior.ai/",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
]

# List to store paths to PDF files found in the 'data' directory
DEFAULT_PDFS = []


# Resolve current directory and 'data' folder path
current_dir = Path(__file__).resolve().parent
data_dir = current_dir / 'data'


# Populate DEFAULT_PDFS with all PDF files in the 'data' directory
if data_dir.exists() and data_dir.is_dir():
    for file_path in data_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            DEFAULT_PDFS.append(str(file_path))  # Add full path to the PDF list


print("Detected URLs:", DEFAULT_URLS)
print("Detected PDF files:", DEFAULT_PDFS)
