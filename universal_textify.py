# universal_textify.py
# Universal OCR post-processor using Ollama with safe fallback.
import subprocess
import re
from datetime import datetime
from typing import Optional

# Strict instruction to produce plain text only.
UNIVERSAL_PROMPT = """
You are a careful document-parsing assistant. Input below is noisy OCR text (it may contain typos, weird spacing, broken lines).
Task:
1) Detect the document type (choose one: RECEIPT, INVOICE, BUSINESS_CARD, REPORT, SPORTS_SCORECARD, OTHER).
2) Correct obvious OCR typos and normalize values (phone numbers, emails, URLs, dates, currency).
3) Produce a CLEAN, HUMAN-READABLE plain-text summary tailored to the detected document type.
4) Output ONLY the plain text summary (no JSON, no markdown, no commentary, no backticks). The summary must use labeled fields (one field per line) and a final RAW: block with the original OCR.

Rules:
- If a field is not found, leave it blank after the label.
- Normalize phone numbers (remove spaces/parentheses/dashes but preserve leading +).
- Lowercase emails.
- Normalize URLs to include https:// when possible.
- Normalize dates to YYYY-MM-DD if possible.
- Normalize currency numbers to plain decimal (e.g., 138.00).
- Items should be a list under ITEMS: using "- " bullet lines if you can find them.
- At the very end include RAW: followed by the original OCR unchanged.
- Output only plain text.

Now extract and produce the plain text summary for the OCR content below.
<<<OCR_TEXT_START>>>
{OCR_TEXT}
<<<OCR_TEXT_END>>>
"""

# Regex helpers for fallback
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')  # coarse
URL_RE = re.compile(r'(https?://[^\s,;]+|www\.[^\s,;]+)', re.IGNORECASE)
DATE_RE = re.compile(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})')
MONEY_RE = re.compile(r'([\$\£\€]?\s?[\d{1,3},]*\d+\.\d{2})')

def normalize_phone(raw: str) -> str:
    digits = re.sub(r'[^+\d]', '', raw)
    if digits.startswith('00'):
        digits = '+' + digits[2:]
    return digits

def normalize_email(e: str) -> str:
    return e.strip().lower()

def normalize_url(u: str) -> str:
    u = u.strip()
    if u.startswith('www.'):
        return 'https://' + u
    if not u.startswith('http'):
        return 'https://' + u
    return u

def detect_type_by_keywords(ocr_text: str) -> str:
    t = ocr_text.lower()
    if any(k in t for k in ['receipt', 'subtotal', 'total', 'tax', 'invoice', 'amount due', 'balance due']):
        return 'RECEIPT'
    if any(k in t for k in ['invoice', 'bill to', 'invoice no', 'invoice number']):
        return 'INVOICE'
    if any(k in t for k in ['business card', 'mobile', 'phone', 'email', '@']) and len(t.splitlines()) < 40:
        return 'BUSINESS_CARD'
    if any(k in t for k in ['report', 'summary', 'executive summary', 'conclusion']):
        return 'REPORT'
    if any(k in t for k in ['wicket', 'innings', 'run', 'overs', 'match', 'score', 'batting', 'bowling']):
        return 'SPORTS_SCORECARD'
    return 'OTHER'

def fallback_build_readable(ocr_text: str) -> str:
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    doc_type = detect_type_by_keywords(ocr_text)
    emails = EMAIL_RE.findall(ocr_text) or []
    phones_raw = PHONE_RE.findall(ocr_text) or []
    phones = [normalize_phone(p) for p in phones_raw]
    urls = [normalize_url(m[0] if isinstance(m, tuple) else m) for m in URL_RE.findall(ocr_text) or []]

    # date
    iso_date = ""
    dates = DATE_RE.findall(ocr_text) or []
    for d in dates:
        try:
            s = d.replace('/', '-')
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%m-%d-%y"):
                try:
                    dt = datetime.strptime(s, fmt)
                    iso_date = dt.strftime("%Y-%m-%d")
                    break
                except:
                    continue
            if iso_date:
                break
        except:
            continue

    # money
    money_matches = MONEY_RE.findall(ocr_text) or []
    total = ""
    if money_matches:
        cleaned = [re.sub(r'[^\d.]', '', m) for m in money_matches]
        try:
            nums = [float(x) for x in cleaned]
            total = "{:.2f}".format(max(nums))
        except:
            total = cleaned[-1] if cleaned else ""

    name = ""
    org = ""
    if doc_type == 'BUSINESS_CARD':
        if lines:
            org = lines[0]
            for cand in lines[1:5]:
                if '@' in cand or re.search(r'\d', cand):
                    continue
                if len(cand.split()) <= 4 and any(ch.isalpha() for ch in cand):
                    name = cand
                    break

    items = []
    for ln in lines:
        if re.search(r'\d+\.\d{2}', ln) or re.search(r'\bx\b', ln, re.IGNORECASE):
            items.append(ln)
    items = list(dict.fromkeys(items))

    parts = []
    parts.append(f"TYPE: {doc_type}")
    if doc_type in ('RECEIPT','INVOICE'):
        org_guess = lines[0] if lines else ""
        parts.append(f"STORE: {org_guess}")
        parts.append(f"DATE: {iso_date}")
        parts.append(f"TOTAL: {total}")
        parts.append("ITEMS:")
        if items:
            for it in items:
                parts.append(f"- {it}")
        else:
            parts.append("")
    elif doc_type == 'BUSINESS_CARD':
        parts.append(f"NAME: {name}")
        parts.append(f"ORG: {org}")
        parts.append(f"PHONE: {', '.join(phones)}")
        parts.append(f"EMAIL: {', '.join([normalize_email(e) for e in emails])}")
        parts.append(f"WEB: {', '.join(urls)}")
        parts.append(f"ADDRESS: ")
    elif doc_type == 'REPORT':
        title = lines[0] if lines else ""
        parts.append(f"TITLE: {title}")
        summary_lines = lines[1:6]
        parts.append("SUMMARY:")
        for s in summary_lines:
            parts.append(f"- {s}")
    elif doc_type == 'SPORTS_SCORECARD':
        parts.append("SCORE SUMMARY:")
        for l in lines[:8]:
            parts.append(f"- {l}")
    else:
        parts.append(f"PHONE: {', '.join(phones)}")
        parts.append(f"EMAIL: {', '.join([normalize_email(e) for e in emails])}")
        parts.append(f"WEB: {', '.join(urls)}")
        parts.append(f"TOTAL: {total}")
    parts.append("")
    parts.append("RAW:")
    parts.append(ocr_text)
    return "\n".join(parts)

def run_ollama_universal(ocr_text: str, model: str = "llama2", timeout: int = 35) -> Optional[str]:
    prompt = UNIVERSAL_PROMPT.replace("{OCR_TEXT}", ocr_text)
    cmd = ["ollama", "run", model, "--no-stream", "--prompt", prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip() or proc.stderr.strip()
        if not out:
            return None
        if "RAW:" not in out:
            out = out.strip() + "\n\nRAW:\n" + ocr_text
        return out
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def universal_textify(ocr_text: str, model: str = "llama2") -> str:
    try:
        out = run_ollama_universal(ocr_text, model=model)
        if out:
            return out
    except Exception:
        pass
    return fallback_build_readable(ocr_text)
