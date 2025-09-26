from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import aiofiles
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import os
import re
import requests  # <-- for Ollama HTTP

# ------------------ OCR libraries ------------------
OCR_LIBS_AVAILABLE = False
try:
    import numpy as np
    from PIL import Image
    import cv2
    import pytesseract
    try:
        import easyocr
    except Exception:
        easyocr = None
    OCR_LIBS_AVAILABLE = True
except Exception:
    np = None
    Image = None
    cv2 = None
    pytesseract = None
    easyocr = None
    OCR_LIBS_AVAILABLE = False

# ------------------ NER import ------------------
try:
    from predictions import getPredictions
    NER_AVAILABLE = True
except Exception:
    getPredictions = None
    NER_AVAILABLE = False

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI()

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# If Tesseract binary is not on PATH (Windows), uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EXECUTOR = ThreadPoolExecutor(max_workers=2)

def base_context(request: Request, title: str):
    return {
        "request": request,
        "title": title,
        "year": datetime.datetime.now().year,
        "site_name": "OCR Model"
    }

# ====================================================
# -------- Ollama Universal Summary Helper -----------
# ====================================================
# Settings
OLLAMA_ENABLED = True
OLLAMA_BASE = "http://127.0.0.1:11434"  # prefer 127.0.0.1 on Windows
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # first call can be slow

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

def ollama_available() -> bool:
    """Quick ping to check the Ollama server."""
    if not OLLAMA_ENABLED:
        return False
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False

def run_ollama_summary(ocr_text: str, model: str = OLLAMA_MODEL, timeout: int = OLLAMA_TIMEOUT) -> Optional[str]:
    """Call the local Ollama server to produce a human-readable summary."""
    if not ollama_available():
        return None
    try:
        payload = {
            "model": model,
            "prompt": UNIVERSAL_PROMPT.replace("{OCR_TEXT}", ocr_text),
            "stream": False
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
        print("Ollama request failed:", e)
        return None

# --------- lightweight fallback summarizer ----------
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')  # loose
URL_RE = re.compile(r'(https?://[^\s,;]+|www\.[^\s,;]+)', re.IGNORECASE)

def fallback_summary(ocr_text: str) -> str:
    """Very simple readable summary if Ollama is unavailable."""
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
        ocr_text
    ]
    return "\n".join(parts)

# ====================================================
# ------------------- ROUTES ------------------------
# ====================================================

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

def preprocess_for_ocr_pil(pil_img: Image.Image,
                           upscale: float = 1.6,
                           bilateral_d: int = 9,
                           bilateral_sigma_color: int = 75,
                           bilateral_sigma_space: int = 75,
                           median_k: int = 3,
                           clahe_clip: float = 3.0,
                           adaptive_block: int = 15,
                           adaptive_c: int = 9,
                           morph_kernel=(2, 2),
                           morph_op: Optional[str] = "open"):
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
    proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, adaptive_c)
    if morph_op:
        kernel = np.ones(morph_kernel, np.uint8)
        if morph_op == 'open':
            proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
        elif morph_op == 'close':
            proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)
    return proc, gray

def run_ocr_sync_on_image(pil_img: Image.Image, proc_np):
    results = {"easy_orig": "", "easy_proc": "", "tess_psm6": "", "tess_psm11": ""}
    try:
        if OCR_LIBS_AVAILABLE and easyocr is not None:
            reader = easyocr.Reader(['en'], gpu=False)
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
        if OCR_LIBS_AVAILABLE and pytesseract is not None:
            pil_for_tess = Image.fromarray(proc_np) if proc_np is not None else pil_img.convert("L")
            results["tess_psm6"] = pytesseract.image_to_string(
                pil_for_tess, lang='eng', config='--oem 3 --psm 6') or ""
            results["tess_psm11"] = pytesseract.image_to_string(
                pil_for_tess, lang='eng', config='--oem 3 --psm 11') or ""
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
    if not (content_type.startswith("image/") or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))):
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

    if not OCR_LIBS_AVAILABLE:
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
        print("Saving summary file failed:", e)
        summary_name = None

    ctx.update({
        "filename": final_name,
        "ocr_source": chosen_key,
        "ocr_text": cleaned_text,
        "summary_file": summary_name,
        "summary_preview": summary_text or ""
    })
    return templates.TemplateResponse("ocr_result.html", ctx)

@app.get("/ocr/download/{fname}")
async def ocr_download(fname: str):
    safe = Path(fname).name
    p = UPLOAD_DIR / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="text/plain", filename=safe)

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
    if not (content_type.startswith("image/") or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))):
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

    ctx.update({
        "filename": safe_name,
        "image_file": annotated_name,
        "text_file": txt_name,
        "entities": entities
    })
    return templates.TemplateResponse("ner_result.html", ctx)

# ---------------- Generic download ----------------
@app.get("/download/{fname}")
async def download(fname: str):
    safe = Path(fname).name
    p = UPLOAD_DIR / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="application/octet-stream", filename=safe)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
