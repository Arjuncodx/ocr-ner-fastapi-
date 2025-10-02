"""
ollama_excel_helper.py

Helper module for Ollama-based text summarization and Excel row/column extraction.
This module provides functions to:
1. Get a summary from Ollama for OCR text
2. Ask Ollama to suggest Excel structure (rows, columns, values)
3. Convert Ollama suggestions into downloadable Excel/CSV files
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration from environment
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Logging setup
logger = logging.getLogger("ollama_excel_helper")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ==================== HTTP Session with Retries ====================
def create_requests_session(retries: int = 3, backoff: float = 1.5) -> requests.Session:
    """
    Create a requests session with automatic retry logic for robustness.
    
    Args:
        retries: Number of retry attempts
        backoff: Backoff factor for exponential retry delay
        
    Returns:
        Configured requests.Session
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ==================== Ollama API Call ====================
def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Call Ollama API with the given prompt and return the response text.
    
    Args:
        prompt: The prompt to send to Ollama
        model: Model name (defaults to OLLAMA_MODEL from env)
        timeout: Request timeout in seconds
        
    Returns:
        Response text from Ollama
        
    Raises:
        RuntimeError: If the API call fails after retries
    """
    model = model or OLLAMA_MODEL
    timeout = timeout or OLLAMA_TIMEOUT
    url = f"{OLLAMA_BASE.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    headers = {"Content-Type": "application/json"}
    session = create_requests_session()
    
    try:
        logger.info(f"Calling Ollama API: model={model}")
        response = session.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract response text from various possible keys
        if "response" in data:
            return str(data["response"]).strip()
        elif "text" in data:
            return str(data["text"]).strip()
        elif "content" in data:
            return str(data["content"]).strip()
        else:
            logger.warning("Unexpected Ollama response format")
            return response.text.strip()
            
    except requests.exceptions.Timeout:
        logger.error(f"Ollama API timeout after {timeout}s")
        raise RuntimeError(f"Ollama API timeout after {timeout}s")
    except requests.exceptions.RequestException as exc:
        logger.error(f"Ollama API request failed: {exc}")
        raise RuntimeError(f"Ollama API request failed: {exc}")
    except Exception as exc:
        logger.exception(f"Unexpected error calling Ollama: {exc}")
        raise RuntimeError(f"Unexpected error calling Ollama: {exc}")


# ==================== Prompt Templates ====================
SUMMARY_PROMPT_TEMPLATE = """You are a document analysis expert. Your task is to read the following OCR-extracted text and provide a clear, concise summary.

Focus on:
- Document type (invoice, receipt, business card, report, etc.)
- Key information (names, dates, amounts, companies)
- Important details that a human would want to know

Provide a 3-7 sentence summary that captures the essence of this document.

OCR TEXT:
{ocr_text}

SUMMARY:"""


EXCEL_STRUCTURE_PROMPT_TEMPLATE = """You are a data structuring expert. Your task is to analyze the following document text and suggest how to organize it into an Excel spreadsheet.

Provide your response as a JSON object with the following structure:
{{
  "suggested_columns": ["Column1", "Column2", "Column3"],
  "rows": [
    {{"Column1": "value1", "Column2": "value2", "Column3": "value3"}},
    {{"Column1": "value4", "Column2": "value5", "Column3": "value6"}}
  ]
}}

Guidelines:
- For invoices/receipts: Use columns like "Item", "Quantity", "Price", "Total"
- For business cards: Use columns like "Field", "Value" with rows for Name, Company, Phone, Email, etc.
- For reports: Extract key data points into a structured table
- Keep column names short and descriptive
- Ensure all rows have the same columns
- Convert dates to YYYY-MM-DD format
- Convert currency to plain decimal numbers (e.g., 1234.56)

DOCUMENT TEXT:
{ocr_text}

OUTPUT (JSON only, no explanations):"""


# ==================== Summary Generation ====================
def generate_summary_from_ollama(
    ocr_text: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Generate a human-readable summary of OCR text using Ollama.
    
    Args:
        ocr_text: Raw OCR text to summarize
        model: Ollama model name
        timeout: Request timeout in seconds
        
    Returns:
        Summary text from Ollama
    """
    if not ocr_text or not ocr_text.strip():
        return "No text provided for summarization."
    
    prompt = SUMMARY_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    
    try:
        summary = call_ollama(prompt, model=model, timeout=timeout)
        return summary if summary else "Summary generation failed."
    except Exception as exc:
        logger.exception(f"Summary generation failed: {exc}")
        return f"Summary generation failed: {str(exc)}"


# ==================== Fallback Summary ====================
def generate_fallback_summary(ocr_text: str) -> str:
    """
    Generate a simple fallback summary when Ollama is unavailable.
    
    Args:
        ocr_text: Raw OCR text
        
    Returns:
        Basic summary extracted using regex patterns
    """
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    
    if not lines:
        return "Empty document - no text extracted."
    
    # Extract basic information
    title = lines[0] if lines else "Document"
    
    # Find emails
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(email_pattern, ocr_text, re.IGNORECASE)
    
    # Find phone numbers
    phone_pattern = r'(\+?\d[\d\-\s().]{6,}\d)'
    phones = re.findall(phone_pattern, ocr_text)
    
    # Find dates
    date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
    dates = re.findall(date_pattern, ocr_text)
    
    # Find currency amounts
    currency_pattern = r'[\$£€]\s?[\d,]+\.?\d{0,2}'
    amounts = re.findall(currency_pattern, ocr_text)
    
    summary_parts = [f"Document Title: {title}"]
    
    if emails:
        summary_parts.append(f"Emails found: {', '.join(emails[:3])}")
    if phones:
        summary_parts.append(f"Phone numbers found: {', '.join(phones[:3])}")
    if dates:
        summary_parts.append(f"Dates found: {', '.join(dates[:3])}")
    if amounts:
        summary_parts.append(f"Amounts found: {', '.join(amounts[:3])}")
    
    summary_parts.append(f"Total lines: {len(lines)}")
    
    return " | ".join(summary_parts)


# ==================== JSON Extraction ====================
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from text that may contain other content.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON as dict, or None if extraction fails
    """
    if not text:
        return None
    
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON code blocks (```json ... ```)
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
    
    # Find balanced braces
    brace_count = 0
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                candidate = text[start_idx:i+1]
                try:
                    return json.loads(candidate)
                except:
                    start_idx = None
    
    return None


# ==================== Excel Structure Generation ====================
def generate_excel_structure_from_ollama(
    ocr_text: str,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ask Ollama to suggest Excel structure (columns and rows) for the OCR text.
    
    Args:
        ocr_text: Raw OCR text to structure
        model: Ollama model name
        timeout: Request timeout in seconds
        
    Returns:
        Dict with 'suggested_columns' and 'rows' keys
    """
    if not ocr_text or not ocr_text.strip():
        return {"suggested_columns": ["Text"], "rows": [{"Text": "No data"}]}
    
    prompt = EXCEL_STRUCTURE_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    
    try:
        response = call_ollama(prompt, model=model, timeout=timeout)
        
        # Try to extract JSON from response
        json_data = extract_json_from_text(response)
        
        if json_data and "suggested_columns" in json_data and "rows" in json_data:
            return json_data
        else:
            logger.warning("Ollama response missing required keys, using fallback")
            return generate_fallback_excel_structure(ocr_text)
            
    except Exception as exc:
        logger.exception(f"Excel structure generation failed: {exc}")
        return generate_fallback_excel_structure(ocr_text)


# ==================== Fallback Excel Structure ====================
def generate_fallback_excel_structure(ocr_text: str) -> Dict[str, Any]:
    """
    Generate a simple fallback Excel structure when Ollama is unavailable or fails.
    
    Args:
        ocr_text: Raw OCR text
        
    Returns:
        Dict with basic structure
    """
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    
    # Try to detect key-value pairs (e.g., "Name: John", "Age: 30")
    key_value_pattern = r'^([A-Za-z\s]+):\s*(.+)$'
    structured_data = []
    
    for line in lines:
        match = re.match(key_value_pattern, line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            structured_data.append({"Field": key, "Value": value})
    
    # If we found structured data, use it
    if structured_data:
        return {
            "suggested_columns": ["Field", "Value"],
            "rows": structured_data
        }
    
    # Otherwise, create a simple line-by-line structure
    return {
        "suggested_columns": ["Line Number", "Text"],
        "rows": [{"Line Number": i+1, "Text": line} for i, line in enumerate(lines[:50])]  # Limit to 50 lines
    }


# ==================== DataFrame Creation ====================
def create_dataframe_from_structure(structure: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Excel structure dict to pandas DataFrame.
    
    Args:
        structure: Dict with 'suggested_columns' and 'rows'
        
    Returns:
        pandas DataFrame
    """
    columns = structure.get("suggested_columns", ["Data"])
    rows = structure.get("rows", [])
    
    if not rows:
        # Return empty DataFrame with columns
        return pd.DataFrame(columns=columns)
    
    # Ensure all rows have all columns
    normalized_rows = []
    for row in rows:
        normalized_row = {col: row.get(col, "") for col in columns}
        normalized_rows.append(normalized_row)
    
    return pd.DataFrame(normalized_rows, columns=columns)


# ==================== File Writing ====================
def write_excel_file(
    df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str = "ocr_data"
) -> Path:
    """
    Write DataFrame to Excel file (.xlsx).
    
    Args:
        df: pandas DataFrame to write
        output_dir: Directory to save the file
        filename_prefix: Prefix for the filename
        
    Returns:
        Path to the created Excel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{filename_prefix}_{timestamp}_{unique_id}.xlsx"
    filepath = output_dir / filename
    
    # Write Excel file with basic formatting
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        # Get the worksheet to apply formatting
        worksheet = writer.sheets['Sheet1']
        
        # Bold header row
        for cell in worksheet[1]:
            cell.font = cell.font.copy(bold=True)
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    logger.info(f"Excel file created: {filepath}")
    return filepath


def write_csv_file(
    df: pd.DataFrame,
    output_dir: Path,
    filename_prefix: str = "ocr_data"
) -> Path:
    """
    Write DataFrame to CSV file.
    
    Args:
        df: pandas DataFrame to write
        output_dir: Directory to save the file
        filename_prefix: Prefix for the filename
        
    Returns:
        Path to the created CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{filename_prefix}_{timestamp}_{unique_id}.csv"
    filepath = output_dir / filename
    
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    logger.info(f"CSV file created: {filepath}")
    return filepath


# ==================== Main Processing Function ====================
def process_ocr_with_ollama(
    ocr_text: str,
    output_dir: Path,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    filename_prefix: str = "ocr_data"
) -> Tuple[str, Path, Path, Dict[str, Any]]:
    """
    Complete processing pipeline:
    1. Generate summary from Ollama
    2. Generate Excel structure from Ollama
    3. Create Excel and CSV files
    
    Args:
        ocr_text: Raw OCR text to process
        output_dir: Directory to save output files
        model: Ollama model name
        timeout: Request timeout in seconds
        filename_prefix: Prefix for output filenames
        
    Returns:
        Tuple of (summary_text, excel_path, csv_path, structure_dict)
    """
    logger.info("Starting OCR processing with Ollama")
    
    # Step 1: Generate summary
    logger.info("Step 1: Generating summary...")
    try:
        summary = generate_summary_from_ollama(ocr_text, model=model, timeout=timeout)
    except Exception as exc:
        logger.warning(f"Ollama summary failed, using fallback: {exc}")
        summary = generate_fallback_summary(ocr_text)
    
    # Step 2: Generate Excel structure
    logger.info("Step 2: Generating Excel structure...")
    try:
        structure = generate_excel_structure_from_ollama(ocr_text, model=model, timeout=timeout)
    except Exception as exc:
        logger.warning(f"Ollama structure generation failed, using fallback: {exc}")
        structure = generate_fallback_excel_structure(ocr_text)
    
    # Step 3: Create DataFrame
    logger.info("Step 3: Creating DataFrame...")
    df = create_dataframe_from_structure(structure)
    
    # Step 4: Write files
    logger.info("Step 4: Writing files...")
    excel_path = write_excel_file(df, output_dir, filename_prefix)
    csv_path = write_csv_file(df, output_dir, filename_prefix)
    
    logger.info("OCR processing with Ollama completed successfully")
    
    return summary, excel_path, csv_path, structure


# ==================== Utility: Check Ollama Availability ====================
def check_ollama_availability() -> bool:
    """
    Check if Ollama server is available and responding.
    
    Returns:
        True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return response.ok
    except:
        return False