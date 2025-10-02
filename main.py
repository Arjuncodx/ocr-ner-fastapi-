#!/usr/bin/env python
# main.py
# FastAPI application for OCR + NER + Ollama-driven Excel export (updated with new Ollama Excel helper)

from __future__ import annotations

import asyncio
import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

# Optional OCR / NER libs (we already check availability below)
try:
    from PIL import Image
    import cv2
    import pytesseract
    try:
        import easyocr
    except Exception:
        easyocr = None
except Exception:
    Image = None
    cv2 = None
    pytesseract = None
    easyocr = None

# Try to import predictions.getPredictions (NER)
try:
    from predictions import getPredictions

    NER_AVAILABLE = True
except Exception:
    getPredictions = None
    NER_AVAILABLE = False

# Try to import the NEW Ollama Excel helper module
try:
    from ollama_excel_helper import (
        process_ocr_with_ollama,
        check_ollama_availability,
        generate_summary_from_ollama,
        generate_fallback_summary,
    )
    OLLAMA_EXCEL_AVAILABLE = True
except Exception as e:
    process_ocr_with_ollama = None
    check_ollama_availability = None
    generate_summary_from_ollama = None
    generate_fallback_summary = None
    OLLAMA_EXCEL_AVAILABLE = False
    logging.warning(f"Ollama Excel helper not available: {e}")

# Try to import the ORIGINAL ocr_to_excel module (keep existing functionality)
try:
    from ocr_to_excel import mount_ocr_to_excel, process_ocr_to_excel

    OCR_TO_EXCEL_AVAILABLE = True
except Exception:
    mount_ocr_to_excel = None
    process_ocr_to_excel = None
    OCR_TO_EXCEL_AVAILABLE = False

# -------------------- App config --------------------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="OCR Model Pro",
    description="Advanced OCR + NER + Ollama AI-powered document processing",
    version="2.0.0"
)
EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Logging
logger = logging.getLogger("main")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Ollama settings (kept here for the original app's helper functions)
OLLAMA_ENABLED = True
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))


def base_context(request: Request, title: str):
    """Generate base template context with common variables."""
    return {
        "request": request,
        "title": title,
        "year": datetime.datetime.now().year,
        "site_name": "OCR Model",
    }


# ------------------ Ollama & fallback helpers (LEGACY - KEEP FOR BACKWARD COMPATIBILITY) ------------------
def ollama_available() -> bool:
    """Check if legacy Ollama endpoint is available."""
    if not OLLAMA_ENABLED:
        return False
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


UNIVERSAL_PROMPT = """
You are an assistant that cleans and organizes noisy OCR text into a neat, human-readable report.
- Correct obvious OCR typos.
- If it looks like a receipt/invoice: show STORE, DATE, TOTAL, ITEMS.
- If it looks like a business card: show NAME, ORG, DES, PHONE(s), EMAIL(s), WEB, ADDRESS.
- Otherwise: provide a short, clear summary.
Always finish with:
RAW:
<original OCR text>
Output only plain text.
OCR TEXT:
{OCR_TEXT}
"""


def run_ollama_summary(ocr_text: str, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Legacy Ollama summary function (kept for backward compatibility)."""
    if not ollama_available():
        return None
    try:
        payload = {
            "model": model,
            "prompt": UNIVERSAL_PROMPT.replace("{OCR_TEXT}", ocr_text),
            "stream": False,
        }
        resp = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            return None
        if "RAW:" not in text:
            text += "\n\nRAW:\n" + ocr_text
        return text
    except Exception as e:
        logger.warning("Ollama summary call failed: %s", e)
        return None


# Lightweight fallback
import re

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s().]{6,}\d)")
URL_RE = re.compile(r"(https?://[^\s,;]+|www\.[^\s,;]+)", re.IGNORECASE)


def fallback_summary(ocr_text: str) -> str:
    """Simple regex-based fallback summary."""
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    title = lines[0] if lines else "Document"
    emails = ", ".join(dict.fromkeys(EMAIL_RE.findall(ocr_text))) or ""
    phones = ", ".join(dict.fromkeys(PHONE_RE.findall(ocr_text))) or ""
    urls = ", ".join(dict.fromkeys(URL_RE.findall(ocr_text))) or ""
    parts = [
        f"TITLE: {title}",
        f"EMAIL: {emails}",
        f"PHONE: {phones}",
        f"WEB: {urls}",
        "",
        "RAW:",
        ocr_text,
    ]
    return "\n".join(parts)


# -------------------- Routes --------------------

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Homepage with PDF upload."""
    ctx = base_context(request, "Home")
    return templates.TemplateResponse("index.html", ctx)


@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """PDF upload endpoint (legacy - kept for compatibility)."""
    filename = file.filename or "uploaded_file.pdf"
    content_type = file.content_type or ""
    if not (filename.lower().endswith(".pdf") or content_type == "application/pdf"):
        ctx = base_context(request, "Upload Error")
        ctx.update({"error": "Only PDF files are allowed."})
        return templates.TemplateResponse("success.html", ctx)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    save_path = UPLOAD_DIR / safe_name

    try:
        async with aiofiles.open(save_path, "wb") as out_file:
            while content := await file.read(1024 * 64):
                await out_file.write(content)
    except Exception as e:
        ctx = base_context(request, "Upload Error")
        ctx.update({"error": f"Saving failed: {e}"})
        return templates.TemplateResponse("success.html", ctx)

    ctx = base_context(request, f"Success — {safe_name}")
    ctx.update({"filename": safe_name})
    return templates.TemplateResponse("success.html", ctx)


# ---------------- Image OCR with NEW Ollama Integration ----------------
@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page(request: Request):
    """OCR upload page."""
    ctx = base_context(request, "OCR Upload")
    return templates.TemplateResponse("ocr.html", ctx)


def preprocess_for_ocr_pil(
    pil_img,
    upscale: float = 1.6,
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 75,
    bilateral_sigma_space: int = 75,
    median_k: int = 3,
    clahe_clip: float = 3.0,
    adaptive_block: int = 15,
    adaptive_c: int = 9,
    morph_kernel=(2, 2),
    morph_op: Optional[str] = "open",
):
    """
    Preprocess image for optimal OCR results.
    Applies: bilateral filter, median blur, upscaling, CLAHE, adaptive threshold, morphology.
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    if median_k and median_k % 2 == 1:
        gray = cv2.medianBlur(gray, median_k)
    if upscale != 1.0:
        new_w = int(gray.shape[1] * upscale)
        new_h = int(gray.shape[0] * upscale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    block = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
    proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, adaptive_c)
    if morph_op:
        kernel = np.ones(morph_kernel, np.uint8)
        if morph_op == "open":
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
        elif morph_op == "close":
            proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)
    return proc, gray


def run_ocr_sync_on_image(pil_img, proc_np):
    """
    Run OCR using multiple engines: EasyOCR and Tesseract with different PSM modes.
    Returns dict with results from each engine.
    """
    results = {"easy_orig": "", "easy_proc": "", "tess_psm6": "", "tess_psm11": ""}
    
    # Try EasyOCR
    try:
        if easyocr is not None:
            reader = easyocr.Reader(["en"], gpu=False)
            arr = np.array(pil_img.convert("RGB"))
            res_orig = reader.readtext(arr, detail=0)
            results["easy_orig"] = "\n".join(res_orig) if res_orig else ""
            if proc_np is not None:
                proc_rgb = cv2.cvtColor(proc_np, cv2.COLOR_GRAY2RGB)
                res_proc = reader.readtext(proc_rgb, detail=0)
                results["easy_proc"] = "\n".join(res_proc) if res_proc else ""
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
    
    # Try Tesseract with different PSM modes
    try:
        if pytesseract is not None:
            pil_for_tess = Image.fromarray(proc_np) if proc_np is not None else pil_img.convert("L")
            results["tess_psm6"] = pytesseract.image_to_string(pil_for_tess, lang="eng", config="--oem 3 --psm 6") or ""
            results["tess_psm11"] = pytesseract.image_to_string(pil_for_tess, lang="eng", config="--oem 3 --psm 11") or ""
    except Exception as e:
        logger.warning(f"Tesseract failed: {e}")
    
    return results


def choose_best_text(results: dict):
    """
    Select the best OCR result based on length and availability.
    Priority: EasyOCR original > EasyOCR processed > Tesseract PSM6 > Tesseract PSM11
    """
    if results.get("easy_orig"):
        return results["easy_orig"], "easy_orig"
    if results.get("easy_proc"):
        return results["easy_proc"], "easy_proc"
    t6 = results.get("tess_psm6", "") or ""
    t11 = results.get("tess_psm11", "") or ""
    return (t6, "tess_psm6") if len(t6) >= len(t11) else (t11, "tess_psm11")


@app.post("/ocr/upload", response_class=HTMLResponse)
async def ocr_upload(request: Request, file: UploadFile = File(...)):
    """
    Main OCR endpoint with NEW Ollama-based summary and Excel generation.
    
    Processing Flow:
    1. Upload image and save to disk
    2. Preprocess image for optimal OCR
    3. Run multi-engine OCR (EasyOCR + Tesseract)
    4. Select best OCR result
    5. Send to Ollama for AI-powered summary
    6. Ask Ollama to suggest Excel structure (rows, columns, values)
    7. Generate downloadable Excel (.xlsx) and CSV files
    8. Return results page with download links
    """
    ctx = base_context(request, "OCR Result")

    filename = file.filename or "uploaded_image"
    content_type = (file.content_type or "").lower()
    if not (content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        ctx.update({"error": "Only image files are allowed for OCR (PNG/JPG/WebP)."})
        return templates.TemplateResponse("ocr_result.html", ctx)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    save_path = UPLOAD_DIR / safe_name

    # Save uploaded file
    try:
        async with aiofiles.open(save_path, "wb") as out_f:
            while chunk := await file.read(1024 * 64):
                await out_f.write(chunk)
    except Exception as e:
        ctx.update({"error": f"Saving failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # Check OCR libraries availability
    if cv2 is None or pytesseract is None:
        ctx.update({"error": "OCR libraries not found. Ensure pillow, pytesseract, opencv-python-headless are installed and tesseract is on PATH."})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # Open and validate image
    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception as e:
        ctx.update({"error": f"Could not open image: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # ========== STEP 1: Perform OCR ==========
    logger.info(f"Starting OCR processing for {safe_name}")
    try:
        loop = asyncio.get_running_loop()
        # Run preprocessing in thread pool to avoid blocking
        proc_np, gray_np = await loop.run_in_executor(EXECUTOR, preprocess_for_ocr_pil, pil_img)
        # Run OCR engines in thread pool
        ocr_results = await loop.run_in_executor(EXECUTOR, run_ocr_sync_on_image, pil_img, proc_np)
    except Exception as e:
        logger.exception(f"OCR processing failed: {e}")
        ctx.update({"error": f"OCR processing failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # Choose best OCR result
    chosen_text, chosen_key = choose_best_text(ocr_results)
    lines = [ln.rstrip() for ln in chosen_text.splitlines() if ln.strip()]
    cleaned_text = "\n".join(lines).strip()

    logger.info(f"OCR completed. Engine: {chosen_key}, Text length: {len(cleaned_text)} chars")

    # Save raw OCR text to file
    final_name = f"{Path(safe_name).stem}_ocr.txt"
    try:
        async with aiofiles.open(UPLOAD_DIR / final_name, "w", encoding="utf-8") as out_f:
            await out_f.write(cleaned_text)
    except Exception as e:
        ctx.update({"error": f"Saving final text failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # ========== STEP 2: Generate Summary with NEW Ollama Helper ==========
    logger.info("Generating Ollama-based summary")
    summary_text = None
    summary_name = None
    
    if OLLAMA_EXCEL_AVAILABLE and generate_summary_from_ollama is not None:
        try:
            # Use new Ollama helper for summary generation
            summary_text = await loop.run_in_executor(
                EXECUTOR,
                generate_summary_from_ollama,
                cleaned_text,
                OLLAMA_MODEL,
                OLLAMA_TIMEOUT
            )
            logger.info("Ollama summary generated successfully")
        except Exception as e:
            logger.warning(f"New Ollama summary failed: {e}, trying fallback")
            # Try new fallback first
            if generate_fallback_summary is not None:
                try:
                    summary_text = generate_fallback_summary(cleaned_text)
                except:
                    summary_text = fallback_summary(cleaned_text)
            else:
                summary_text = fallback_summary(cleaned_text)
    else:
        # Use legacy Ollama summary if new helper not available
        logger.info("Using legacy Ollama summary (new helper not available)")
        summary_text = run_ollama_summary(cleaned_text, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)
        if summary_text is None:
            summary_text = fallback_summary(cleaned_text)

    # Save summary to file
    if summary_text:
        summary_name = f"{Path(final_name).stem}_summary.txt"
        try:
            async with aiofiles.open(UPLOAD_DIR / summary_name, "w", encoding="utf-8") as sf:
                await sf.write(summary_text)
            logger.info(f"Summary saved to {summary_name}")
        except Exception as e:
            logger.warning(f"Saving summary file failed: {e}")
            summary_name = None

    # ========== STEP 3: Generate Excel/CSV with NEW Ollama Helper ==========
    logger.info("Generating Excel and CSV files with Ollama structure suggestions")
    csv_name = None
    xlsx_name = None
    excel_structure = None

    if OLLAMA_EXCEL_AVAILABLE and process_ocr_with_ollama is not None:
        try:
            # Use new comprehensive Ollama processing pipeline
            # This will ask Ollama: "What rows, columns, and values should I put in Excel for this text?"
            _, excel_path, csv_path, structure = await loop.run_in_executor(
                EXECUTOR,
                process_ocr_with_ollama,
                cleaned_text,
                UPLOAD_DIR,
                OLLAMA_MODEL,
                OLLAMA_TIMEOUT,
                Path(safe_name).stem
            )
            
            if excel_path and excel_path.exists():
                xlsx_name = excel_path.name
                logger.info(f"Excel file generated: {xlsx_name}")
            
            if csv_path and csv_path.exists():
                csv_name = csv_path.name
                logger.info(f"CSV file generated: {csv_name}")
            
            excel_structure = structure
            logger.info(f"Excel structure: {len(structure.get('suggested_columns', []))} columns, {len(structure.get('rows', []))} rows")
            
        except Exception as e:
            logger.exception(f"New Ollama Excel generation failed: {e}")
            # Ensure values remain None so template won't show broken links
            xlsx_name = None
            csv_name = None
            excel_structure = None

    # ========== FALLBACK: Simple CSV if Ollama Excel generation failed ==========
    if not csv_name and not xlsx_name:
        logger.info("Using fallback CSV generation (line-by-line)")
        try:
            import csv
            csv_name = f"{Path(final_name).stem}_fallback.csv"
            with open(UPLOAD_DIR / csv_name, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["Line Number", "Text"])
                for idx, ln in enumerate(lines, 1):
                    writer.writerow([idx, ln])
            logger.info(f"Fallback CSV created: {csv_name}")
        except Exception as e:
            logger.warning(f"Failed to create fallback CSV: {e}")
            csv_name = None

    # ========== STEP 4: Return Results to Template ==========
    ctx.update(
        {
            "filename": final_name,
            "ocr_source": chosen_key,
            "ocr_text": cleaned_text,
            "summary_file": summary_name,
            "summary_preview": summary_text or "",
            "csv_file": csv_name,
            "xlsx_file": xlsx_name,
            "excel_structure": excel_structure,  # Pass structure info for display in template
        }
    )
    
    logger.info(f"OCR processing complete for {safe_name}")
    return templates.TemplateResponse("ocr_result.html", ctx)


# ---------------- NER (NO CHANGES - KEEPING ORIGINAL FUNCTIONALITY) ----------------
@app.get("/ner", response_class=HTMLResponse)
async def ner_page(request: Request):
    """NER upload page - unchanged."""
    ctx = base_context(request, "NER Upload")
    return templates.TemplateResponse("ner.html", ctx)


@app.post("/ner/upload", response_class=HTMLResponse)
async def ner_upload(request: Request, file: UploadFile = File(...)):
    """
    NER processing endpoint - completely unchanged.
    Extracts named entities using custom spaCy model.
    """
    ctx = base_context(request, "NER Result")

    filename = file.filename or "uploaded_image"
    content_type = (file.content_type or "").lower()
    if not (content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        ctx.update({"error": "Only image files are allowed for NER (PNG/JPG/WebP)."})
        return templates.TemplateResponse("ner_result.html", ctx)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    save_path = UPLOAD_DIR / safe_name

    try:
        async with aiofiles.open(save_path, "wb") as out_f:
            while chunk := await file.read(1024 * 64):
                await out_f.write(chunk)
    except Exception as e:
        ctx.update({"error": f"Saving failed: {e}"})
        return templates.TemplateResponse("ner_result.html", ctx)

    if not NER_AVAILABLE or getPredictions is None:
        ctx.update({"error": "NER model not available. Ensure predictions.py and output/model-best/ exist and spaCy is installed."})
        return templates.TemplateResponse("ner_result.html", ctx)

    try:
        with open(save_path, "rb") as f:
            file_bytes = f.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None.")
    except Exception as e:
        ctx.update({"error": f"Could not open uploaded image: {e}"})
        return templates.TemplateResponse("ner_result.html", ctx)

    try:
        loop = asyncio.get_running_loop()
        img_bb, entities = await loop.run_in_executor(EXECUTOR, getPredictions, img)
    except Exception as e:
        ctx.update({"error": f"NER processing failed: {e}"})
        return templates.TemplateResponse("ner_result.html", ctx)

    annotated_name = f"{Path(filename).stem}_{timestamp}_ner.png"
    txt_name = f"{Path(filename).stem}_{timestamp}_ner.txt"

    try:
        cv2.imwrite(str(UPLOAD_DIR / annotated_name), img_bb)
    except Exception as e:
        ctx.update({"error": f"Saving annotated image failed: {e}"})
        return templates.TemplateResponse("ner_result.html", ctx)

    try:
        lines = []
        for key, items in entities.items():
            lines.append(f"{key}: {', '.join(items) if items else ''}")
        async with aiofiles.open(UPLOAD_DIR / txt_name, "w", encoding="utf-8") as out_f:
            await out_f.write("\n".join(lines))
    except Exception as e:
        ctx.update({"error": f"Saving entities text failed: {e}"})
        return templates.TemplateResponse("ner_result.html", ctx)

    ctx.update(
        {
            "filename": safe_name,
            "image_file": annotated_name,
            "text_file": txt_name,
            "entities": entities,
        }
    )
    return templates.TemplateResponse("ner_result.html", ctx)


# ---------------- Generic download endpoint ----------------
@app.get("/download/{fname}")
async def download(fname: str):
    """
    Generic file download endpoint.
    Supports: .xlsx, .csv, .txt, .png, .jpg, .jpeg, .pdf
    """
    safe = Path(fname).name
    p = UPLOAD_DIR / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Set appropriate media type based on file extension
    if safe.lower().endswith(".xlsx"):
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif safe.lower().endswith(".csv"):
        media = "text/csv"
    elif safe.lower().endswith(".txt"):
        media = "text/plain"
    elif safe.lower().endswith(".png"):
        media = "image/png"
    elif safe.lower().endswith((".jpg", ".jpeg")):
        media = "image/jpeg"
    elif safe.lower().endswith(".pdf"):
        media = "application/pdf"
    else:
        media = "application/octet-stream"
    
    return FileResponse(str(p), media_type=media, filename=safe)


# Special download endpoint for OCR results (alias for compatibility)
@app.get("/ocr/download/{fname}")
async def download_ocr_result(fname: str):
    """Special endpoint for OCR text files - redirects to main download."""
    return await download(fname)


# ----------------- Mount original ocr_to_excel router if available (BACKWARD COMPATIBILITY) -----------------
if OCR_TO_EXCEL_AVAILABLE and mount_ocr_to_excel is not None:
    try:
        mount_ocr_to_excel(app)
        logger.info("✓ Original ocr_to_excel router mounted at /ocr/process-to-excel")
    except Exception as exc:
        logger.exception("Failed to mount ocr_to_excel router: %s", exc)
else:
    logger.info("⚠ Original ocr_to_excel not available; /ocr/process-to-excel endpoint disabled")


# ----------------- Health Check Endpoint -----------------
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status and available features.
    Returns status of all major components.
    """
    ollama_status = "unavailable"
    if OLLAMA_EXCEL_AVAILABLE and check_ollama_availability:
        try:
            ollama_status = "available" if check_ollama_availability() else "unavailable"
        except:
            ollama_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "features": {
            "ocr_opencv_tesseract": cv2 is not None and pytesseract is not None,
            "ocr_easyocr": easyocr is not None,
            "ner_spacy": NER_AVAILABLE,
            "ollama_excel_helper": OLLAMA_EXCEL_AVAILABLE,
            "original_ocr_to_excel": OCR_TO_EXCEL_AVAILABLE,
        },
        "ollama": {
            "status": ollama_status,
            "base_url": OLLAMA_BASE,
            "model": OLLAMA_MODEL,
        },
        "upload_directory": str(UPLOAD_DIR),
    }


# ----------------- Application Startup Info -----------------
@app.on_event("startup")
async def startup_event():
    """Log startup information about available features."""
    logger.info("="*70)
    logger.info("🚀 OCR Model Pro Server Starting")
    logger.info("="*70)
    logger.info(f"📁 Upload Directory: {UPLOAD_DIR}")
    logger.info(f"🔧 Thread Pool Workers: {EXECUTOR._max_workers}")
    logger.info("")
    logger.info("📊 Feature Status:")
    logger.info(f"  {'✓' if cv2 and pytesseract else '✗'} OpenCV + Tesseract OCR")
    logger.info(f"  {'✓' if easyocr else '✗'} EasyOCR")
    logger.info(f"  {'✓' if NER_AVAILABLE else '✗'} NER (spaCy Model)")
    logger.info(f"  {'✓' if OLLAMA_EXCEL_AVAILABLE else '✗'} Ollama Excel Helper (NEW)")
    logger.info(f"  {'✓' if OCR_TO_EXCEL_AVAILABLE else '✗'} Original OCR-to-Excel Module")
    logger.info("")
    logger.info("🤖 Ollama Configuration:")
    logger.info(f"  Base URL: {OLLAMA_BASE}")
    logger.info(f"  Model: {OLLAMA_MODEL}")
    logger.info(f"  Timeout: {OLLAMA_TIMEOUT}s")
    
    if OLLAMA_EXCEL_AVAILABLE and check_ollama_availability:
        try:
            is_available = check_ollama_availability()
            logger.info(f"  Status: {'✓ Available' if is_available else '✗ Unavailable'}")
        except:
            logger.info(f"  Status: ✗ Error checking availability")
    else:
        logger.info(f"  Status: ⚠ Helper not loaded")
    
    logger.info("="*70)
    logger.info("🌐 Server ready at http://127.0.0.1:8000")
    logger.info("📚 API docs at http://127.0.0.1:8000/docs")
    logger.info("💚 Health check at http://127.0.0.1:8000/health")
    logger.info("="*70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down OCR Model Pro Server")
    EXECUTOR.shutdown(wait=True)
    logger.info("✓ Thread pool executor shutdown complete")


# ----------------- Additional Utility Endpoints -----------------

@app.get("/api/status")
async def api_status():
    """
    API endpoint to check system status (JSON response).
    Useful for monitoring and integration.
    """
    return {
        "service": "OCR Model Pro",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "capabilities": {
            "ocr": {
                "tesseract": pytesseract is not None,
                "easyocr": easyocr is not None,
                "opencv": cv2 is not None,
            },
            "ner": {
                "available": NER_AVAILABLE,
                "entities": ["NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"] if NER_AVAILABLE else [],
            },
            "ai": {
                "ollama_excel": OLLAMA_EXCEL_AVAILABLE,
                "ollama_base": OLLAMA_BASE,
                "ollama_model": OLLAMA_MODEL,
            },
        },
    }


@app.get("/api/models")
async def list_models():
    """
    List available models and their status.
    """
    models = {
        "ocr_engines": [],
        "ner_model": None,
        "ollama_model": None,
    }
    
    if pytesseract is not None:
        models["ocr_engines"].append({
            "name": "Tesseract",
            "type": "ocr",
            "status": "available",
            "modes": ["psm6", "psm11"],
        })
    
    if easyocr is not None:
        models["ocr_engines"].append({
            "name": "EasyOCR",
            "type": "ocr",
            "status": "available",
            "languages": ["en"],
        })
    
    if NER_AVAILABLE:
        models["ner_model"] = {
            "name": "Custom spaCy NER",
            "type": "ner",
            "status": "available",
            "entities": ["NAME", "ORG", "DES", "PHONE", "EMAIL", "WEB"],
            "model_path": "output/model-best",
        }
    
    if OLLAMA_EXCEL_AVAILABLE:
        ollama_available_status = "unavailable"
        if check_ollama_availability:
            try:
                ollama_available_status = "available" if check_ollama_availability() else "unavailable"
            except:
                ollama_available_status = "error"
        
        models["ollama_model"] = {
            "name": OLLAMA_MODEL,
            "type": "llm",
            "status": ollama_available_status,
            "base_url": OLLAMA_BASE,
            "capabilities": ["summarization", "excel_structure", "data_extraction"],
        }
    
    return models


@app.post("/api/ocr/quick")
async def quick_ocr_api(file: UploadFile = File(...)):
    """
    Quick OCR API endpoint that returns JSON instead of HTML.
    Useful for programmatic access.
    
    Returns:
        JSON with OCR text, summary (if Ollama available), and file paths
    """
    if cv2 is None or pytesseract is None:
        raise HTTPException(
            status_code=503,
            detail="OCR libraries not available"
        )
    
    # Validate file type
    content_type = (file.content_type or "").lower()
    filename = file.filename or "image"
    if not (content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed (PNG/JPG/WebP)"
        )
    
    # Save file
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    save_path = UPLOAD_DIR / safe_name
    
    try:
        async with aiofiles.open(save_path, "wb") as out_f:
            while chunk := await file.read(1024 * 64):
                await out_f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Open image
    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    # Perform OCR
    try:
        loop = asyncio.get_running_loop()
        proc_np, _ = await loop.run_in_executor(EXECUTOR, preprocess_for_ocr_pil, pil_img)
        ocr_results = await loop.run_in_executor(EXECUTOR, run_ocr_sync_on_image, pil_img, proc_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")
    
    chosen_text, chosen_key = choose_best_text(ocr_results)
    lines = [ln.rstrip() for ln in chosen_text.splitlines() if ln.strip()]
    cleaned_text = "\n".join(lines).strip()
    
    # Save OCR text
    text_filename = f"{Path(safe_name).stem}_ocr.txt"
    text_path = UPLOAD_DIR / text_filename
    try:
        async with aiofiles.open(text_path, "w", encoding="utf-8") as out_f:
            await out_f.write(cleaned_text)
    except:
        text_filename = None
    
    # Generate summary if Ollama available
    summary = None
    if OLLAMA_EXCEL_AVAILABLE and generate_summary_from_ollama:
        try:
            summary = await loop.run_in_executor(
                EXECUTOR,
                generate_summary_from_ollama,
                cleaned_text,
                OLLAMA_MODEL,
                OLLAMA_TIMEOUT
            )
        except:
            pass
    
    return {
        "success": True,
        "image_file": safe_name,
        "ocr_engine": chosen_key,
        "text": cleaned_text,
        "text_file": text_filename,
        "summary": summary,
        "lines_count": len(lines),
        "char_count": len(cleaned_text),
        "download_urls": {
            "image": f"/download/{safe_name}",
            "text": f"/download/{text_filename}" if text_filename else None,
        },
    }


# ----------------- Run Uvicorn (dev) -----------------
if __name__ == "__main__":
    import uvicorn
    
    # Log initial configuration
    logger.info("")
    logger.info("="*70)
    logger.info("🔧 OCR Model Pro - Configuration")
    logger.info("="*70)
    logger.info(f"Python: {os.sys.version.split()[0]}")
    logger.info(f"FastAPI: Starting web server")
    logger.info(f"Upload Directory: {UPLOAD_DIR}")
    logger.info("")
    logger.info("📦 Available Libraries:")
    logger.info(f"  {'✓' if cv2 else '✗'} OpenCV")
    logger.info(f"  {'✓' if pytesseract else '✗'} Tesseract")
    logger.info(f"  {'✓' if easyocr else '✗'} EasyOCR")
    logger.info(f"  {'✓' if Image else '✗'} Pillow")
    logger.info(f"  {'✓' if NER_AVAILABLE else '✗'} spaCy (NER)")
    logger.info("")
    logger.info("🤖 AI Features:")
    logger.info(f"  {'✓' if OLLAMA_EXCEL_AVAILABLE else '✗'} Ollama Excel Helper (NEW)")
    logger.info(f"  {'✓' if OCR_TO_EXCEL_AVAILABLE else '✗'} Original OCR-to-Excel")
    logger.info("")
    logger.info("⚙️ Ollama Settings:")
    logger.info(f"  Base URL: {OLLAMA_BASE}")
    logger.info(f"  Model: {OLLAMA_MODEL}")
    logger.info(f"  Timeout: {OLLAMA_TIMEOUT}s")
    logger.info("="*70)
    logger.info("")
    
    # Start server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )