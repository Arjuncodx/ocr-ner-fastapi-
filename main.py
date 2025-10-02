#!/usr/bin/env python
# main.py
# FastAPI application for OCR + NER + Ollama-driven Excel export (updated to produce CSV/XLSX)

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

# Try to import the ocr_to_excel module we added (process_ocr_to_excel)
try:
    from ocr_to_excel import mount_ocr_to_excel, process_ocr_to_excel  # type: ignore

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

app = FastAPI()
EXECUTOR = ThreadPoolExecutor(max_workers=2)

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
    return {
        "request": request,
        "title": title,
        "year": datetime.datetime.now().year,
        "site_name": "OCR Model",
    }


# ------------------ Ollama & fallback helpers ------------------
def ollama_available() -> bool:
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
    ctx = base_context(request, "Home")
    return templates.TemplateResponse("index.html", ctx)


@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
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


# ---------------- Image OCR ----------------
@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page(request: Request):
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
    results = {"easy_orig": "", "easy_proc": "", "tess_psm6": "", "tess_psm11": ""}
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
    except Exception:
        pass
    try:
        if pytesseract is not None:
            pil_for_tess = Image.fromarray(proc_np) if proc_np is not None else pil_img.convert("L")
            results["tess_psm6"] = pytesseract.image_to_string(pil_for_tess, lang="eng", config="--oem 3 --psm 6") or ""
            results["tess_psm11"] = pytesseract.image_to_string(pil_for_tess, lang="eng", config="--oem 3 --psm 11") or ""
    except Exception:
        pass
    return results


def choose_best_text(results: dict):
    if results.get("easy_orig"):
        return results["easy_orig"], "easy_orig"
    if results.get("easy_proc"):
        return results["easy_proc"], "easy_proc"
    t6 = results.get("tess_psm6", "") or ""
    t11 = results.get("tess_psm11", "") or ""
    return (t6, "tess_psm6") if len(t6) >= len(t11) else (t11, "tess_psm11")


@app.post("/ocr/upload", response_class=HTMLResponse)
async def ocr_upload(request: Request, file: UploadFile = File(...)):
    ctx = base_context(request, "OCR Result")

    filename = file.filename or "uploaded_image"
    content_type = (file.content_type or "").lower()
    if not (content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
        ctx.update({"error": "Only image files are allowed for OCR (PNG/JPG/WebP)."})
        return templates.TemplateResponse("ocr_result.html", ctx)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = f"{Path(filename).stem}_{timestamp}{Path(filename).suffix}"
    save_path = UPLOAD_DIR / safe_name

    try:
        async with aiofiles.open(save_path, "wb") as out_f:
            while chunk := await file.read(1024 * 64):
                await out_f.write(chunk)
    except Exception as e:
        ctx.update({"error": f"Saving failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    if cv2 is None or pytesseract is None:
        ctx.update({"error": "OCR libraries not found. Ensure pillow, pytesseract, opencv-python-headless are installed and tesseract is on PATH."})
        return templates.TemplateResponse("ocr_result.html", ctx)

    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception as e:
        ctx.update({"error": f"Could not open image: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    try:
        loop = asyncio.get_running_loop()
        proc_np, gray_np = await loop.run_in_executor(EXECUTOR, preprocess_for_ocr_pil, pil_img)
        ocr_results = await loop.run_in_executor(EXECUTOR, run_ocr_sync_on_image, pil_img, proc_np)
    except Exception as e:
        ctx.update({"error": f"OCR processing failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    chosen_text, chosen_key = choose_best_text(ocr_results)
    lines = [ln.rstrip() for ln in chosen_text.splitlines() if ln.strip()]
    cleaned_text = "\n".join(lines).strip()

    final_name = f"{Path(safe_name).stem}_ocr.txt"
    try:
        async with aiofiles.open(UPLOAD_DIR / final_name, "w", encoding="utf-8") as out_f:
            await out_f.write(cleaned_text)
    except Exception as e:
        ctx.update({"error": f"Saving final text failed: {e}"})
        return templates.TemplateResponse("ocr_result.html", ctx)

    # ---------- NEW: Summary via Ollama (with fallback) ----------
    summary_text = run_ollama_summary(cleaned_text, model=OLLAMA_MODEL, timeout=OLLAMA_TIMEOUT)
    if summary_text is None:
        summary_text = fallback_summary(cleaned_text)

    summary_name = f"{Path(final_name).stem}_summary.txt"
    try:
        async with aiofiles.open(UPLOAD_DIR / summary_name, "w", encoding="utf-8") as sf:
            await sf.write(summary_text)
    except Exception as e:
        logger.warning("Saving summary file failed: %s", e)
        summary_name = None

    # ---------- NEW: Generate CSV/XLSX using process_ocr_to_excel if available ----------
    csv_name = None
    xlsx_name = None
    if OCR_TO_EXCEL_AVAILABLE and process_ocr_to_excel is not None:
        try:
            # process_ocr_to_excel returns (path_to_excel, excel_rows)
            excel_path, excel_rows = await asyncio.get_running_loop().run_in_executor(
                EXECUTOR, lambda: process_ocr_to_excel(cleaned_text, model=OLLAMA_MODEL, out_dir=UPLOAD_DIR)
            )
            if excel_path and Path(excel_path).exists():
                xlsx_name = Path(excel_path).name
                # also write CSV fallback for first sheet
                try:
                    import pandas as pd

                    # If excel has data, try to write first sheet as CSV (best-effort)
                    with pd.ExcelFile(str(excel_path)) as reader:
                        sheet_names = reader.sheet_names
                        if sheet_names:
                            df0 = pd.read_excel(reader, sheet_name=sheet_names[0])
                            csv_name = f"{Path(excel_path).stem}_{sheet_names[0]}.csv"
                            df0.to_csv(UPLOAD_DIR / csv_name, index=False)
                except Exception as e:
                    logger.warning("Could not write CSV fallback: %s", e)
        except Exception as e:
            logger.exception("Excel generation failed: %s", e)
            # ensure values remain None so template won't show links
            xlsx_name = None
            csv_name = None

    # if process_ocr_to_excel not available, optionally build a simple CSV fallback using regex/extracted lines
    if not csv_name and not xlsx_name:
        try:
            # attempt to create a simple CSV with raw lines as a single column
            import csv

            csv_name = f"{Path(final_name).stem}_fallback.csv"
            with open(UPLOAD_DIR / csv_name, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(["line"])
                for ln in lines:
                    writer.writerow([ln])
        except Exception as e:
            logger.warning("Failed to create fallback CSV: %s", e)
            csv_name = None

    ctx.update(
        {
            "filename": final_name,
            "ocr_source": chosen_key,
            "ocr_text": cleaned_text,
            "summary_file": summary_name,
            "summary_preview": summary_text or "",
            "csv_file": csv_name,
            "xlsx_file": xlsx_name,
        }
    )
    return templates.TemplateResponse("ocr_result.html", ctx)


# ---------------- NER ----------------
@app.get("/ner", response_class=HTMLResponse)
async def ner_page(request: Request):
    ctx = base_context(request, "NER Upload")
    return templates.TemplateResponse("ner.html", ctx)


@app.post("/ner/upload", response_class=HTMLResponse)
async def ner_upload(request: Request, file: UploadFile = File(...)):
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


# ---------------- Generic download ----------------
@app.get("/download/{fname}")
async def download(fname: str):
    safe = Path(fname).name
    p = UPLOAD_DIR / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Use FileResponse so browser can download
    # set media_type for xlsx, csv, txt appropriately for browser convenience
    if safe.lower().endswith(".xlsx"):
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif safe.lower().endswith(".csv"):
        media = "text/csv"
    elif safe.lower().endswith(".txt"):
        media = "text/plain"
    else:
        media = "application/octet-stream"
    return FileResponse(str(p), media_type=media, filename=safe)


# ----------------- Mount ocr_to_excel router if available -----------------
if OCR_TO_EXCEL_AVAILABLE and mount_ocr_to_excel is not None:
    try:
        mount_ocr_to_excel(app)
        logger.info("ocr_to_excel router mounted at /ocr/process-to-excel")
    except Exception as exc:
        logger.exception("Failed to mount ocr_to_excel router: %s", exc)
else:
    logger.info("ocr_to_excel not available; /ocr/process-to-excel endpoint disabled (place ocr_to_excel.py next to main.py).")


# ----------------- Run Uvicorn (dev) -----------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
