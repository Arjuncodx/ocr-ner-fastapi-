
"""
ocr_to_excel.py

End-to-end helper to:
 - send raw OCR text to Ollama for cleaning & executive summary,
 - ask Ollama for an "excel_rows" mapping (sheet,row,column,value,value_type),
 - convert mappings to DataFrames and write an .xlsx file,
 - expose a FastAPI router for /ocr/process-to-excel.

Environment variables:
 - OLLAMA_BASE (default: http://127.0.0.1:11434)
 - OLLAMA_MODEL (default: qwen3:8b)
 - OLLAMA_TIMEOUT (seconds, default: 300)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional imports for image handling (used by FastAPI endpoint)
try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

# -- configuration from env --
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Uploads directory (same as your app)
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("ocr_to_excel")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Constants
OLLAMA_GENERATE_PATH = "/api/generate"
MAX_CHUNK_SIZE = 28000  # characters (safe prompt chunking)
JSON_REQUIRED_KEYS_STEP_A = {"cleaned_text", "executive_summary", "fields", "raw_text"}
JSON_REQUIRED_KEYS_STEP_B = {"excel_rows"}


@dataclass
class ExcelRow:
    sheet: str
    row: int
    column: str  # column header name
    value: Any
    value_type: str  # string|number|date|currency|phone|email|url|boolean

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sheet": self.sheet,
            "row": int(self.row),
            "column": self.column,
            "value": self.value,
            "value_type": self.value_type,
        }


# ----------------- HTTP helper -----------------
def requests_session_with_retries(
    retries: int = 3, backoff_factor: float = 1.5, status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504)
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["POST", "GET", "PUT", "DELETE", "HEAD", "OPTIONS"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def call_ollama(prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> str:
    """
    Call local Ollama HTTP API /api/generate and return the raw textual response.
    Retries and structured logging included.
    """
    model = model or OLLAMA_MODEL
    timeout = timeout or OLLAMA_TIMEOUT
    url = OLLAMA_BASE.rstrip("/") + OLLAMA_GENERATE_PATH
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    sess = requests_session_with_retries()
    try:
        logger.debug("Calling Ollama %s (timeout=%s)", model, timeout)
        resp = sess.post(url, json=payload, timeout=timeout, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Ollama variants may store text at data['response'] or nested; try common keys
        if isinstance(data, dict) and "response" in data:
            return str(data["response"])
        if isinstance(data, dict) and "outputs" in data:
            # sometimes outputs -> list of dicts with 'content' or 'output'
            outputs = data["outputs"]
            if isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, dict):
                    for k in ("content", "text", "output"):
                        if k in first:
                            return str(first[k])
                return json.dumps(outputs)
        # fallback: raw text
        text = resp.text
        return text
    except Exception as exc:
        logger.exception("Ollama call failed: %s", exc)
        raise RuntimeError(f"Ollama request failed: {exc}")


# ----------------- JSON extraction helpers -----------------
def _find_json_in_text(text: str) -> Optional[str]:
    """
    Heuristic: find first outer JSON object in text.
    Tries to find balanced braces that produce valid JSON.
    """
    if not text:
        return None
    # Try simple direct parse first
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Search for {...} blocks and try to parse
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        # continue searching
                        start_idx = None
                        continue
    return None


def _extract_and_validate_json(raw_text: str, required_keys: set) -> Dict[str, Any]:
    """
    Try to get a JSON object from raw text that contains the required keys.
    Throws a ValueError if it cannot be recovered.
    """
    if not raw_text:
        raise ValueError("Empty response from model")

    # First try straight parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and required_keys.issubset(set(parsed.keys())):
            return parsed
    except Exception:
        pass

    # Try to find a JSON block inside text
    candidate = _find_json_in_text(raw_text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and required_keys.issubset(set(parsed.keys())):
                return parsed
        except Exception:
            # Fall through to failure
            pass

    # As a last attempt, try to find smaller JSON blocks and merge (not ideal)
    raise ValueError("Could not extract valid JSON with required keys from model output.")


# ----------------- Fallback summarizer -----------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s().]{6,}\d)")
MONEY_RE = re.compile(r"[\$£€]\s?[0-9,]+\.\d{2}")
URL_RE = re.compile(r"(https?://[^\s,;]+|www\.[^\s,;]+)", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")


def fallback_clean_and_summary(ocr_text: str) -> Dict[str, Any]:
    """
    Lightweight fallback when Ollama is unavailable or returns invalid output.
    Returns a dict matching Step A schema (best-effort).
    """
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    title = lines[0] if lines else ""
    emails = list(dict.fromkeys(EMAIL_RE.findall(ocr_text)))
    phones = list(dict.fromkeys(PHONE_RE.findall(ocr_text)))
    urls = list(dict.fromkeys(URL_RE.findall(ocr_text)))
    money = MONEY_RE.findall(ocr_text)
    date = ""
    date_matches = DATE_RE.findall(ocr_text)
    if date_matches:
        date = date_matches[0]
    fields = {
        "type": "OTHER",
        "store": title if "receipt" in ocr_text.lower() or "invoice" in ocr_text.lower() else "",
        "date": date,
        "total": money[-1] if money else "",
        "items": [],
        "name": "",
        "org": "",
        "phones": phones,
        "emails": [e.lower() for e in emails],
        "web": urls,
        "address": "",
    }
    return {
        "cleaned_text": "\n".join(lines),
        "executive_summary": f"Fallback summary of document: {title}",
        "fields": fields,
        "raw_text": ocr_text,
    }


# ----------------- Prompt templates -----------------
STEP_A_PROMPT = """
You are a strict JSON-only responder. Input is noisy OCR text. Your job:
1) Clean obvious OCR typos, normalize phone/email/url/date/currency formatting.
2) Produce a short executive summary describing document type and important fields.
3) Output a JSON object with these exact keys: cleaned_text, executive_summary, fields, raw_text

The "fields" object must contain these keys exactly:
type, store, date, total, items, name, org, phones, emails, web, address

- type must be one of RECEIPT, INVOICE, BUSINESS_CARD, REPORT, SPORTS_SCORECARD, OTHER.
- cleaned_text: cleaned and normalized plain text (string).
- executive_summary: 1-4 short sentences.
- items must be an array of strings (line items or empty array).
- phones, emails, web: arrays (may be empty).
- date must be YYYY-MM-DD when possible else empty string.
- total must be a decimal string with two decimals like "138.00" or empty string.

Output ONLY a single JSON object (no explanation, no Markdown).

OCR_INPUT:
<<<START>>>
{ocr_text}
<<<END>>>
"""

STEP_B_PROMPT = """
You are a strict JSON-only responder. Input is a cleaned document and executive summary (JSON) from previous step.
Your job: produce a JSON object with a single key "excel_rows" whose value is an array of objects of this exact shape:
{
  "sheet": "<sheet name, default 'Sheet1'>",
  "row": <positive integer>,
  "column": "<column header name>",  # header name string, NOT Excel letter
  "value": <string|number|boolean>,
  "value_type": "<string|number|date|currency|phone|email|url|boolean>"
}

Rules:
 - Header row MUST be row=1 and include column names.
 - Data rows start at row=2 and have the same columns as header.
 - Normalize: dates -> YYYY-MM-DD, currency -> plain decimal with 2 decimals, phone -> digits with + if present, emails -> lowercase, urls -> include https://
 - No extra text. Output only: {"excel_rows": [ ... ]}

Input JSON:
{clean_summary}
"""

# ----------------- Higher level functions -----------------


def _chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = 200) -> List[str]:
    """
    Split text into roughly max_size chunks with small overlap.
    """
    if len(text) <= max_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(start + max_size - overlap, end)
    return chunks


def clean_and_summarize(ocr_text: str, model: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Step A: send OCR text to Ollama for cleaning and executive summary.
    Returns the parsed JSON dict following the STEP A schema. Raises on unrecoverable errors.
    Falls back to internal regex-based summarizer if Ollama fails.
    """
    if not ocr_text or not ocr_text.strip():
        raise ValueError("Empty OCR text provided.")

    model = model or OLLAMA_MODEL
    timeout = timeout or OLLAMA_TIMEOUT

    # Chunking for very long texts:
    chunks = _chunk_text(ocr_text, max_size=MAX_CHUNK_SIZE)
    if len(chunks) == 1:
        prompt = STEP_A_PROMPT.format(ocr_text=ocr_text)
        try:
            raw = call_ollama(prompt, model=model, timeout=timeout)
            parsed = _extract_and_validate_json(raw, JSON_REQUIRED_KEYS_STEP_A)
            return parsed
        except Exception as exc:
            logger.warning("Ollama Step A failed: %s. Falling back.", exc)
            return fallback_clean_and_summary(ocr_text)
    # If many chunks, first get per-chunk cleaned_text summaries then merge them and call once final
    per_chunk_cleaned = []
    for idx, ch in enumerate(chunks):
        prompt = STEP_A_PROMPT.format(ocr_text=ch)
        try:
            raw = call_ollama(prompt, model=model, timeout=timeout)
            parsed = _extract_and_validate_json(raw, JSON_REQUIRED_KEYS_STEP_A)
            per_chunk_cleaned.append(parsed["cleaned_text"])
        except Exception as exc:
            logger.warning("Ollama chunk %s failed: %s", idx, exc)
            # fallback to chunk lines if Ollama fails on chunk
            per_chunk_cleaned.append("\n".join([ln.strip() for ln in ch.splitlines() if ln.strip()]))
    merged_text = "\n\n".join(per_chunk_cleaned)
    # now call final pass to produce unified schema
    final_prompt = STEP_A_PROMPT.format(ocr_text=merged_text)
    try:
        raw = call_ollama(final_prompt, model=model, timeout=timeout)
        parsed = _extract_and_validate_json(raw, JSON_REQUIRED_KEYS_STEP_A)
        return parsed
    except Exception as exc:
        logger.warning("Final Ollama Step A failed: %s. Using fallback.", exc)
        return fallback_clean_and_summary(ocr_text)


def generate_excel_rows(clean_summary: Dict[str, Any], model: Optional[str] = None, timeout: Optional[int] = None) -> List[ExcelRow]:
    """
    Step B: ask Ollama to produce excel_rows mapping. Returns list[ExcelRow].
    """
    model = model or OLLAMA_MODEL
    timeout = timeout or OLLAMA_TIMEOUT

    # Prepare a compact JSON string of the cleaned summary to pass into prompt
    clean_json = json.dumps(clean_summary, ensure_ascii=False, indent=2)
    prompt = STEP_B_PROMPT.format(clean_summary=clean_json)

    try:
        raw = call_ollama(prompt, model=model, timeout=timeout)
        parsed = _extract_and_validate_json(raw, JSON_REQUIRED_KEYS_STEP_B)
    except Exception as exc:
        logger.exception("Step B Ollama failed: %s", exc)
        raise RuntimeError("Failed to obtain excel_rows from Ollama: %s" % exc)

    excel_rows_raw = parsed.get("excel_rows", [])
    if not isinstance(excel_rows_raw, list):
        raise ValueError("Invalid excel_rows format from model (expected list).")

    rows: List[ExcelRow] = []
    for r in excel_rows_raw:
        # Validate each entry
        if not all(k in r for k in ("sheet", "row", "column", "value", "value_type")):
            logger.warning("Skipping malformed excel row: %s", r)
            continue
        try:
            rownum = int(r["row"])
        except Exception:
            logger.warning("Invalid row number; skipping: %s", r)
            continue
        rows.append(
            ExcelRow(
                sheet=str(r.get("sheet") or "Sheet1"),
                row=rownum,
                column=str(r["column"]),
                value=r["value"],
                value_type=str(r["value_type"]),
            )
        )
    if not rows:
        logger.warning("No valid excel rows returned by Ollama.")
    return rows


# ----------------- Convert rows -> DataFrame / Excel -----------------
def rows_to_dataframes(rows: List[ExcelRow]) -> Dict[str, pd.DataFrame]:
    """
    Convert list of ExcelRow into a dict of sheet -> pandas.DataFrame
    The scheme: header row is row==1 entries which define column headers. Data rows (row>=2)
    are assembled using those header names. If missing headers, we derive columns from union of names.
    """
    if not rows:
        return {}
    # Group by sheet
    sheets: Dict[str, List[ExcelRow]] = {}
    for r in rows:
        sheets.setdefault(r.sheet, []).append(r)

    dfs: Dict[str, pd.DataFrame] = {}
    for sheet, items in sheets.items():
        # Build header map
        headers = [r for r in items if r.row == 1]
        header_names = []
        if headers:
            # sort by appearance
            header_names = [h.column for h in headers]
        else:
            # fallback: unique column names from items
            header_names = sorted({h.column for h in items})
        # gather data rows
        data_rows_map: Dict[int, Dict[str, Any]] = {}
        for r in items:
            if r.row == 1:
                continue
            data_rows_map.setdefault(r.row, {})
            data_rows_map[r.row][r.column] = _coerce_value(r.value, r.value_type)
        # convert to list sorted by row index
        if data_rows_map:
            rows_sorted = []
            for rownum in sorted(data_rows_map.keys()):
                rowdict = {col: data_rows_map[rownum].get(col, "") for col in header_names}
                rows_sorted.append(rowdict)
            df = pd.DataFrame(rows_sorted, columns=header_names)
        else:
            df = pd.DataFrame(columns=header_names)
        dfs[sheet] = df
    return dfs


def _coerce_value(value: Any, value_type: str) -> Any:
    """
    Try to coerce typed values into python objects for DataFrame. Keep graceful fallback.
    """
    if value is None:
        return ""
    vt = (value_type or "").lower()
    try:
        if vt == "number" or vt == "currency":
            # currency strings like "138.00" -> float
            if isinstance(value, (int, float)):
                return value
            s = str(value).replace(",", "").strip()
            return float(s)
        if vt == "date":
            # attempt iso parse
            s = str(value).strip()
            # If value already YYYY-MM-DD, keep string; pandas will parse if asked
            return s
        if vt in ("phone", "email", "url", "string"):
            return str(value)
        if vt == "boolean":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            return s in ("1", "true", "yes", "y")
    except Exception:
        pass
    return value


def write_excel_from_rows(rows: List[ExcelRow], out_path: Path) -> Path:
    """
    Create an .xlsx file containing all sheets derived from rows.
    Returns the path to the written file.
    """
    dfs = rows_to_dataframes(rows)
    if not dfs:
        # create an empty workbook with message
        df = pd.DataFrame({"Info": ["No data"]})
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        return out_path

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            # sanitize sheet name
            safe_sheet = str(sheet_name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe_sheet)
    return out_path


# ----------------- End-to-end helper -----------------
def process_ocr_to_excel(
    ocr_text: str, model: Optional[str] = None, out_dir: Optional[Path] = None, filename_prefix: str = "ocr_excel"
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Given raw OCR text, perform Step A and Step B and write an Excel file.
    Returns (path_to_excel, excel_rows_as_dicts)
    """
    model = model or OLLAMA_MODEL
    out_dir = out_dir or UPLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    unique = uuid.uuid4().hex[:8]
    out_filename = f"{filename_prefix}_{timestamp}_{unique}.xlsx"
    out_path = out_dir / out_filename

    summary = clean_and_summarize(ocr_text, model=model)
    rows = generate_excel_rows(summary, model=model)
    write_excel_from_rows(rows, out_path)
    # return rows as dicts for response
    return out_path, [r.as_dict() for r in rows]


# ----------------- FastAPI router mount -----------------
def mount_ocr_to_excel(app) -> None:
    """
    Mounts an APIRouter onto the FastAPI app with the /ocr/process-to-excel endpoint.
    Usage:
      from ocr_to_excel import mount_ocr_to_excel
      mount_ocr_to_excel(app)
    """
    try:
        from fastapi import APIRouter, File, Form, UploadFile, Request, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
    except Exception as exc:
        logger.exception("FastAPI not installed or import failed: %s", exc)
        raise

    router = APIRouter()

    @router.post("/ocr/process-to-excel")
    async def _process_to_excel(request: Request, ocr_text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
        """
        Accepts either:
         - form field `ocr_text` (raw OCR string), or
         - multipart file `file` (image). If image provided, we extract text with pytesseract.
        Returns:
         - JSON: {"excel_rows": [...], "excel_path": "<filename>"} on success
         - or returns the Excel file directly if the client sets Accept header to application/octet-stream.
        """
        # choose input
        if not ocr_text and not file:
            raise HTTPException(status_code=400, detail="Provide either form field 'ocr_text' or upload an image file as 'file'.")

        if file:
            # read bytes and use pytesseract if available
            content = await file.read()
            if Image is None or pytesseract is None:
                raise HTTPException(status_code=500, detail="Image OCR not available on server (Pillow/pytesseract missing).")
            try:
                img = Image.open(BytesIO(content))
                # convert to RGB (robust)
                img = img.convert("RGB")
                extracted = pytesseract.image_to_string(img, lang="eng")
                ocr_text_local = extracted or ""
            except Exception as exc:
                logger.exception("Failed to extract text from uploaded image: %s", exc)
                raise HTTPException(status_code=400, detail=f"Image OCR failed: {exc}")
        else:
            ocr_text_local = ocr_text or ""

        # Run processing
        try:
            path_to_excel, excel_rows = process_ocr_to_excel(ocr_text_local, model=OLLAMA_MODEL, out_dir=UPLOAD_DIR)
        except Exception as exc:
            logger.exception("Processing failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

        # If client expects file download, return FileResponse
        accept = request.headers.get("accept", "")
        if "application/octet-stream" in accept or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in accept:
            return FileResponse(str(path_to_excel), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=path_to_excel.name)

        return JSONResponse({"excel_rows": excel_rows, "excel_path": f"/download/{path_to_excel.name}"})

    app.include_router(router)
