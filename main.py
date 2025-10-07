#!/usr/bin/env python
"""
main.py - OCR ELITE SYSTEM v15.1 ENHANCED ENTITY EXTRACTION
============================================================
IMPROVEMENTS:
✅ Multi-pass Ollama extraction (catches missing fields)
✅ Enhanced prompt engineering for better accuracy
✅ Smart document structure analysis
✅ Intelligent field mapping with fallbacks
✅ All previous features preserved

Senior Python OCR Developer - Fortune 500 Grade
October 2025
"""
from __future__ import annotations

import asyncio
import base64
import csv
import datetime
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from collections import OrderedDict, defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Union

# Core imports
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# Ollama
try:
    from ollama_client_optimized import OptimizedOllamaClient, OllamaConfig
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OptimizedOllamaClient = None
    OllamaConfig = None

# PaddleOCR
try:
    from paddle_ocr_engine import PaddleOCREngine
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCREngine = None

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
CACHE_DIR = BASE_DIR / "cache"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, CACHE_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_ultimate_hybrid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI(
    title="OCR Elite v15.1 Enhanced Extraction",
    description="Ollama AI + Regex + Multi-pass Extraction",
    version="15.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Config
MAX_FILE_SIZE = 50 * 1024 * 1024
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
EXECUTOR = ThreadPoolExecutor(max_workers=12)

logger.info("="*80)
logger.info("🚀 OCR ELITE v15.1 ENHANCED ENTITY EXTRACTION - INITIALIZING")
logger.info("="*80)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class OCREngineResult:
    """Single OCR engine result"""
    engine: str
    text: str
    confidence: float
    processing_time: float
    success: bool
    error: Optional[str] = None
    lines_detected: int = 0

@dataclass
class VotingOCRResult:
    """Multi-engine voting result"""
    final_text: str
    best_engine: str
    engine_confidence: float
    quality_score: float
    engines_used: List[str]
    individual_results: Dict[str, OCREngineResult]
    correction_applied: bool = False
    processing_time: float = 0.0

@dataclass
class CleanedTextResult:
    """Ollama cleaned text result"""
    original_text: str
    cleaned_text: str
    corrections_made: int
    confidence: float

@dataclass
class SmartExcelStructure:
    """AI-suggested Excel structure with enhanced fields"""
    document_type: str
    columns: List[str]
    values: Dict[str, str]
    confidence: float
    extraction_method: str  # 'ollama', 'regex', 'hybrid', or 'enhanced'
    extraction_success: bool = True
    missing_fields: List[str] = field(default_factory=list)  # NEW: Track missing fields
    extraction_passes: int = 1  # NEW: Track how many passes were done

@dataclass
class ProcessingResult:
    """Complete processing result"""
    job_id: str
    status: str
    ocr_text: str
    cleaned_text: str
    ocr_confidence: float
    quality_score: float
    ocr_engines: List[str]
    best_engine: str
    summary: str
    entities: Dict[str, List[str]]
    excel_structure: Optional[SmartExcelStructure]
    processing_time: float
    excel_report: Optional[str] = None
    json_report: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# FILE CONVERTER (UNCHANGED)
# ============================================================================
class UniversalFileConverter:
    """Convert ANY image format"""
    
    @staticmethod
    def convert_to_png(input_path: Path) -> Path:
        try:
            if not PIL_AVAILABLE:
                return input_path
            
            if input_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                return input_path
            
            logger.info(f"🔄 Converting {input_path.suffix}...")
            img = Image.open(input_path)
            
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            output_path = TEMP_DIR / f"{input_path.stem}_converted.png"
            img.save(output_path, 'PNG', quality=95)
            return output_path
        except:
            return input_path

# ============================================================================
# IMAGE PREPROCESSING (UNCHANGED)
# ============================================================================
class AdvancedImagePreprocessor:
    """7-stage image preprocessing"""
    
    @staticmethod
    def preprocess(image_path: Path) -> Optional[np.ndarray]:
        try:
            if not CV2_AVAILABLE:
                return None
            
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            coords = np.column_stack(np.where(morph > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) > 0.5:
                    h, w = morph.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                    morph = cv2.warpAffine(morph, M, (w, h), 
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
            
            final = cv2.medianBlur(morph, 3)
            return final
        except:
            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                pass
            return None

# ============================================================================
# VOTING SYSTEM (UNCHANGED)
# ============================================================================
class UltimateVotingSystem:
    """3-engine voting with proper scoring"""
    
    def __init__(self):
        logger.info("✅ Voting system initialized")
    
    def calculate_quality_score(self, results: Dict[str, OCREngineResult], winner: OCREngineResult) -> float:
        """Calculate overall quality score"""
        try:
            if not results:
                return 0.0
            
            avg_confidence = sum(r.confidence for r in results.values()) / len(results)
            text_lengths = [len(r.text) for r in results.values()]
            avg_length = sum(text_lengths) / len(text_lengths)
            length_score = min(avg_length / 100, 1.0)
            
            if len(results) > 1:
                max_len = max(len(r.text) for r in results.values())
                min_len = min(len(r.text) for r in results.values())
                agreement_score = min_len / max_len if max_len > 0 else 1.0
            else:
                agreement_score = 1.0
            
            quality = (avg_confidence * 0.30 + 
                      winner.confidence * 0.35 + 
                      length_score * 0.15 + 
                      agreement_score * 0.20)
            
            return min(quality, 1.0)
        except:
            return 0.7
    
    def vote(self, easy: OCREngineResult, tess: OCREngineResult, paddle: OCREngineResult) -> VotingOCRResult:
        """Vote for best engine"""
        try:
            logger.info("\n🗳️  3-ENGINE VOTING")
            start = time.time()
            
            results = {}
            if easy.success and easy.text and len(easy.text.strip()) > 5:
                results['easyocr'] = easy
                logger.info(f"  ✅ EasyOCR: {easy.confidence:.2%}")
            
            if tess.success and tess.text and len(tess.text.strip()) > 5:
                results['tesseract'] = tess
                logger.info(f"  ✅ Tesseract: {tess.confidence:.2%}")
            
            if paddle.success and paddle.text and len(paddle.text.strip()) > 5:
                results['paddleocr'] = paddle
                logger.info(f"  ✅ PaddleOCR: {paddle.confidence:.2%}")
            
            if not results:
                return VotingOCRResult("", "none", 0.0, 0.0, [], {}, False, time.time()-start)
            
            winner_name = max(results.items(), key=lambda x: x[1].confidence)[0]
            winner = results[winner_name]
            quality = self.calculate_quality_score(results, winner)
            
            logger.info(f"🏆 WINNER: {winner_name.upper()}")
            logger.info(f"   Confidence: {winner.confidence:.2%}")
            logger.info(f"   Quality: {quality:.2%}")
            
            return VotingOCRResult(
                final_text=winner.text,
                best_engine=winner_name,
                engine_confidence=winner.confidence,
                quality_score=quality,
                engines_used=list(results.keys()),
                individual_results=results,
                correction_applied=False,
                processing_time=time.time()-start
            )
        except:
            return VotingOCRResult("", "error", 0.0, 0.0, [], {}, False, 0.0)

# ============================================================================
# ADVANCED REGEX EXTRACTOR (UNCHANGED - BACKUP SYSTEM)
# ============================================================================
class AdvancedRegexExtractor:
    """Advanced regex extraction as intelligent backup"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Comprehensive regex patterns"""
        return {
            'dates': [
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
            ],
            'amounts': [
                r'\$\s*\d+[,.]?\d*\.?\d{2}',
                r'\d+[,.]\d{2}',
                r'USD\s*\d+[,.]\d{2}',
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
            ],
            'emails': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phones': [
                r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
            ],
            'invoice_numbers': [
                r'(?:Invoice|Receipt|Bill|Order)\s*#?\s*:?\s*([A-Z0-9-]+)',
                r'#\s*([A-Z0-9-]{5,})',
                r'\b[A-Z]{2,}\d{5,}\b'
            ],
            'account_numbers': [
                r'(?:Account|Acct|A/C)\s*#?\s*:?\s*(\d{5,})',
                r'\b\d{10,}\b'
            ],
            'employee_codes': [  # NEW: Added employee code patterns
                r'(?:ICNO|Employee Code|Employee ID|Staff ID|EMP ID)\s*:?\s*(\d{5,})',
                r'\b\d{6}\b',  # 6-digit codes
                r'\bIC\d{5,}\b'
            ],
            'names': [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',  # Full names
                r'(?:Name|Employee)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
            ],
            'months': [r'(?:Month)\s*:?\s*(\d{1,2})', r'\b(0?[1-9]|1[0-2])\b'],
            'years': [r'(?:Year)\s*:?\s*(\d{4})', r'\b(19|20)\d{2}\b'],
            'addresses': [
                r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
            ],
            'totals': [r'(?:Total|Grand Total|Amount Due)\s*:?\s*\$?\s*(\d+[,.]\d{2})'],
        }
    
    def identify_document_type(self, text: str) -> str:
        """Identify document type"""
        text_lower = text.lower()
        
        keywords = {
            'Pay Slip': ['pay slip', 'payslip', 'salary', 'net amount payable'],
            'Invoice': ['invoice', 'bill to', 'ship to'],
            'Receipt': ['receipt', 'thank you'],
            'Bill': ['bill', 'account', 'due date'],
            'Statement': ['statement', 'balance'],
        }
        
        scores = {doc: sum(1 for kw in kws if kw in text_lower) for doc, kws in keywords.items()}
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'Document'
    
    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """Extract all entities"""
        results = {}
        for entity_type, patterns in self.patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found.extend([m if isinstance(m, str) else ''.join(m) for m in matches])
            results[entity_type] = list(set(found))[:10]
        return results
    
    def extract_smart_fields(self, text: str, doc_type: str) -> Dict[str, str]:
        """Extract fields intelligently"""
        entities = self.extract_all(text)
        fields = {}
        
        # Extract based on document type
        if entities.get('employee_codes'):
            fields['Employee Code (ICNO)'] = entities['employee_codes'][0]
        
        if entities.get('names'):
            fields['Employee Name'] = entities['names'][0]
        
        if entities.get('months'):
            fields['Month'] = entities['months'][0]
        
        if entities.get('years'):
            fields['Year'] = entities['years'][0]
        
        if entities.get('dates'):
            fields['Date'] = entities['dates'][0]
        
        if entities.get('amounts'):
            fields['Amount'] = entities['amounts'][0]
            if len(entities['amounts']) > 1:
                fields['Total'] = entities['amounts'][-1]
        
        if entities.get('invoice_numbers'):
            fields['Document ID'] = entities['invoice_numbers'][0]
        
        if entities.get('addresses'):
            fields['Address'] = entities['addresses'][0]
        
        return fields if fields else {'Date': 'Not Found', 'Amount': 'Not Found'}
# ============================================================================
# ENHANCED OLLAMA AI PROCESSOR - MULTI-PASS EXTRACTION
# ============================================================================
class EnhancedOllamaAIProcessor:
    """Enhanced AI processor with multi-pass extraction for maximum accuracy"""
    
    def __init__(self, ollama_client: OptimizedOllamaClient):
        self.ollama = ollama_client
        self.regex_extractor = AdvancedRegexExtractor()
        logger.info("✅ Enhanced Ollama AI Processor initialized with multi-pass extraction")
    
    async def clean_ocr_text(self, raw_text: str) -> CleanedTextResult:
        """Clean OCR text (typos, spelling) - UNCHANGED"""
        try:
            if not raw_text or len(raw_text.strip()) < 5:
                return CleanedTextResult(raw_text, raw_text, 0, 0.0)
            
            prompt = f"""You are an OCR text correction expert. Fix spelling mistakes, typos, and OCR errors in this text.
Keep the original meaning and structure. Only fix clear errors.

Original text:
{raw_text[:2000]}

Return ONLY the corrected text. Do not add explanations."""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=60.0
            )
            
            cleaned = str(response.get("response", raw_text) if isinstance(response, dict) else response)
            cleaned = cleaned.strip()
            
            if not cleaned or len(cleaned) < len(raw_text) * 0.3:
                cleaned = raw_text
            
            corrections = sum(1 for a, b in zip(raw_text[:1000], cleaned[:1000]) if a != b)
            confidence = 0.9 if corrections > 0 else 0.7
            
            logger.info(f"✅ Text cleaned: {corrections} corrections")
            return CleanedTextResult(raw_text, cleaned, corrections, confidence)
        
        except asyncio.TimeoutError:
            logger.warning("⚠️  Text cleaning timeout - using original")
            return CleanedTextResult(raw_text, raw_text, 0, 0.5)
        except Exception as e:
            logger.warning(f"⚠️  Text cleaning error: {e}")
            return CleanedTextResult(raw_text, raw_text, 0, 0.3)
    
    async def identify_document_type_ai(self, text: str) -> str:
        """AI-powered document type identification"""
        try:
            prompt = f"""Identify the document type from this text. Reply with ONLY ONE of these types:
- Pay Slip
- Invoice
- Receipt
- Bill
- Statement
- Form
- Report
- Letter
- Other

Text:
{text[:500]}

Document type:"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=30.0
            )
            
            doc_type = str(response.get("response", "Document") if isinstance(response, dict) else response)
            doc_type = doc_type.strip().split('\n')[0]
            
            valid_types = ['Pay Slip', 'Invoice', 'Receipt', 'Bill', 'Statement', 'Form', 'Report', 'Letter']
            for vt in valid_types:
                if vt.lower() in doc_type.lower():
                    return vt
            
            return "Document"
        
        except:
            return self.regex_extractor.identify_document_type(text)
    
    async def extract_entities_multipass(self, text: str, doc_type: str) -> Dict[str, str]:
        """
        🆕 MULTI-PASS ENTITY EXTRACTION - THE GAME CHANGER!
        
        Strategy:
        1. First Pass: Comprehensive field detection
        2. Second Pass: Fill missing critical fields
        3. Third Pass: Validate and cross-check
        """
        logger.info("\n🎯 MULTI-PASS ENTITY EXTRACTION STARTING")
        
        # PASS 1: Comprehensive Initial Extraction
        logger.info("📋 PASS 1: Comprehensive field detection")
        entities_pass1 = await self._extraction_pass_1_comprehensive(text, doc_type)
        
        logger.info(f"   ✅ Pass 1 extracted {len(entities_pass1)} fields")
        
        # Identify missing critical fields
        missing_fields = self._identify_missing_critical_fields(entities_pass1, doc_type)
        
        if missing_fields:
            logger.info(f"   ⚠️  Missing critical fields: {', '.join(missing_fields)}")
            
            # PASS 2: Targeted extraction for missing fields
            logger.info("📋 PASS 2: Targeted missing field extraction")
            entities_pass2 = await self._extraction_pass_2_targeted(text, missing_fields)
            
            # Merge results
            entities_pass1.update(entities_pass2)
            logger.info(f"   ✅ Pass 2 filled {len(entities_pass2)} missing fields")
        
        # PASS 3: Validation and refinement
        logger.info("📋 PASS 3: Validation and refinement")
        final_entities = await self._extraction_pass_3_validate(text, entities_pass1, doc_type)
        
        logger.info(f"🏆 FINAL: Extracted {len(final_entities)} total fields")
        
        return final_entities
    
    async def _extraction_pass_1_comprehensive(self, text: str, doc_type: str) -> Dict[str, str]:
        """Pass 1: Comprehensive extraction with enhanced prompt"""
        try:
            # Enhanced prompt with better instructions
            prompt = f"""You are a document analysis expert. Extract ALL information from this {doc_type}.

CRITICAL INSTRUCTIONS:
1. Find EVERY piece of information in the document
2. Use EXACT field names as they appear
3. Extract ALL numbers, codes, names, dates, amounts
4. If you see "ICNO:", extract it as "Employee Code (ICNO)"
5. If you see "Month:", extract it as "Month"
6. Don't skip any field even if you're not 100% sure

Document text:
{text[:2500]}

Return in this EXACT format (one per line):
Field Name: Exact Value

Example:
Document ID: 12345
Employee Code (ICNO): 160192
Month: 6
Year: 2016
Employee Name: Shri Prathap Simha
State: Karnataka

Now extract ALL fields from the document above:"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=90.0
            )
            
            result_text = str(response.get("response", "") if isinstance(response, dict) else response)
            
            # Parse results
            entities = {}
            for line in result_text.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        
                        # Clean up key and value
                        key = re.sub(r'^[-*•\d.]+\s*', '', key)  # Remove bullets/numbers
                        
                        if val and len(val) < 200 and val.lower() not in ['na', 'n/a', 'null', 'none', 'not found', 'not available']:
                            entities[key] = val
            
            return entities
        
        except Exception as e:
            logger.warning(f"⚠️  Pass 1 extraction error: {e}")
            return {}
    
    async def _extraction_pass_2_targeted(self, text: str, missing_fields: List[str]) -> Dict[str, str]:
        """Pass 2: Targeted extraction for specific missing fields"""
        try:
            fields_str = ", ".join(missing_fields)
            
            prompt = f"""You are a document analysis expert. Find these SPECIFIC missing fields in the document:

MISSING FIELDS TO FIND:
{fields_str}

Document text:
{text[:2000]}

INSTRUCTIONS:
1. Search carefully for each missing field
2. Look for similar terms (e.g., "ICNO" = "Employee Code", "IC NO" = "ICNO")
3. Extract the value even if the label is slightly different
4. For numeric codes (like ICNO), look for 5-6 digit numbers
5. For names, look for capitalized words after "Name" or "Employee"

Return ONLY the found fields in this format:
Field Name: Value

Example:
Employee Code (ICNO): 160192
Month: 6"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=60.0
            )
            
            result_text = str(response.get("response", "") if isinstance(response, dict) else response)
            
            # Parse results
            entities = {}
            for line in result_text.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        
                        if val and val.lower() not in ['na', 'n/a', 'null', 'none', 'not found']:
                            entities[key] = val
            
            # Also try regex as backup for this pass
            regex_results = self.regex_extractor.extract_smart_fields(text, "Document")
            for field in missing_fields:
                if field not in entities:
                    for regex_key, regex_val in regex_results.items():
                        if field.lower() in regex_key.lower() or regex_key.lower() in field.lower():
                            entities[field] = regex_val
                            break
            
            return entities
        
        except Exception as e:
            logger.warning(f"⚠️  Pass 2 extraction error: {e}")
            # Fallback to regex
            return self.regex_extractor.extract_smart_fields(text, "Document")
    
    async def _extraction_pass_3_validate(self, text: str, entities: Dict[str, str], doc_type: str) -> Dict[str, str]:
        """Pass 3: Validate and refine extracted entities"""
        try:
            # Quick validation pass
            validated = {}
            
            for key, val in entities.items():
                # Skip empty or invalid values
                if not val or val.lower() in ['na', 'n/a', 'null', 'none', 'not found', 'not available']:
                    continue
                
                # Skip very long values (likely errors)
                if len(val) > 300:
                    continue
                
                # Clean up value
                val = val.strip()
                val = re.sub(r'\s+', ' ', val)  # Normalize whitespace
                
                validated[key] = val
            
            return validated
        
        except:
            return entities
    
    def _identify_missing_critical_fields(self, entities: Dict[str, str], doc_type: str) -> List[str]:
        """Identify which critical fields are missing based on document type"""
        critical_fields = {
            'Pay Slip': [
                'Employee Code (ICNO)',
                'ICNO',
                'Employee Code',
                'Employee Name',
                'Name',
                'Month',
                'Year',
                'Net Amount Payable',
                'Total',
                'State'
            ],
            'Invoice': [
                'Invoice Number',
                'Document ID',
                'Date',
                'Total',
                'Amount',
                'Customer Name'
            ],
            'Receipt': [
                'Receipt Number',
                'Date',
                'Amount',
                'Total'
            ],
            'Bill': [
                'Bill Number',
                'Date',
                'Amount Due',
                'Total'
            ],
        }
        
        # Get critical fields for this document type
        required = critical_fields.get(doc_type, [])
        
        # Find missing fields
        missing = []
        for field in required:
            found = False
            for existing_key in entities.keys():
                if field.lower() in existing_key.lower() or existing_key.lower() in field.lower():
                    found = True
                    break
            if not found:
                missing.append(field)
        
        return missing[:5]  # Limit to top 5 missing fields
    
    async def suggest_excel_columns_ai(self, text: str, doc_type: str) -> List[str]:
        """AI suggests Excel columns - ENHANCED"""
        try:
            prompt = f"""You are a document analysis expert. Suggest Excel column names for this {doc_type}.

Document text:
{text[:1000]}

INSTRUCTIONS:
1. Suggest 8-15 relevant column names
2. Use clear, professional names
3. Include common fields like: Document ID, Date, Amount, Name, etc.
4. For Pay Slips, ALWAYS include: Employee Code (ICNO), Employee Name, Month, Year, State
5. Return ONLY column names, one per line

Column names:"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=45.0
            )
            
            result = str(response.get("response", "") if isinstance(response, dict) else response)
            
            columns = []
            for line in result.split('\n'):
                line = line.strip()
                line = re.sub(r'^[-*•\d.]+\s*', '', line)  # Remove bullets
                if line and len(line) < 50 and line not in columns:
                    columns.append(line)
            
            # Ensure minimum columns
            if len(columns) < 5:
                columns = ['Document ID', 'Date', 'Amount', 'Name', 'Description', 'Total']
            
            return columns[:15]
        
        except:
            return ['Document ID', 'Date', 'Amount', 'Name', 'Description', 'Total']
    
    async def build_smart_excel_structure_enhanced(self, text: str) -> SmartExcelStructure:
        """
        🆕 ENHANCED EXCEL STRUCTURE BUILDER - MAIN IMPROVEMENT!
        Uses multi-pass extraction for maximum accuracy
        """
        try:
            logger.info("\n🏗️  BUILDING ENHANCED EXCEL STRUCTURE")
            
            # Step 1: Identify document type
            doc_type = await self.identify_document_type_ai(text)
            logger.info(f"   📄 Document Type: {doc_type}")
            
            # Step 2: Multi-pass entity extraction (THE KEY IMPROVEMENT!)
            entities = await self.extract_entities_multipass(text, doc_type)
            
            # Step 3: AI suggest columns based on extracted entities
            suggested_columns = await self.suggest_excel_columns_ai(text, doc_type)
            
            # Step 4: Map entities to columns intelligently
            final_columns = []
            final_values = {}
            
            # Use extracted entity keys as columns
            for key in entities.keys():
                if key not in final_columns:
                    final_columns.append(key)
                    final_values[key] = entities[key]
            
            # Add suggested columns that don't exist
            for col in suggested_columns:
                if col not in final_columns:
                    final_columns.append(col)
                    # Try to find value from entities
                    found_value = None
                    for ent_key, ent_val in entities.items():
                        if col.lower() in ent_key.lower() or ent_key.lower() in col.lower():
                            found_value = ent_val
                            break
                    final_values[col] = found_value if found_value else "Not found (no unique identifier is provided)"
            
            # Check for missing fields
            missing_fields = [col for col, val in final_values.items() 
                            if not val or val == "Not found (no unique identifier is provided)"]
            
            confidence = 1.0 - (len(missing_fields) / len(final_columns)) if final_columns else 0.0
            
            logger.info(f"   ✅ Extracted {len(entities)} entities")
            logger.info(f"   ✅ Created {len(final_columns)} columns")
            logger.info(f"   ⚠️  {len(missing_fields)} fields missing")
            logger.info(f"   📊 Confidence: {confidence:.2%}")
            
            return SmartExcelStructure(
                document_type=doc_type,
                columns=final_columns,
                values=final_values,
                confidence=confidence,
                extraction_method='enhanced_multipass',
                extraction_success=True,
                missing_fields=missing_fields,
                extraction_passes=3
            )
        
        except Exception as e:
            logger.error(f"❌ Enhanced structure building error: {e}")
            logger.error(traceback.format_exc())
            
            # Fallback to regex
            return await self._fallback_excel_structure(text)
    
    async def _fallback_excel_structure(self, text: str) -> SmartExcelStructure:
        """Fallback to regex extraction"""
        try:
            logger.info("   🔄 Using regex fallback extraction")
            doc_type = self.regex_extractor.identify_document_type(text)
            entities = self.regex_extractor.extract_smart_fields(text, doc_type)
            
            columns = list(entities.keys())
            return SmartExcelStructure(
                document_type=doc_type,
                columns=columns,
                values=entities,
                confidence=0.6,
                extraction_method='regex_fallback',
                extraction_success=True,
                missing_fields=[],
                extraction_passes=1
            )
        except:
            return SmartExcelStructure(
                document_type="Document",
                columns=["Field", "Value"],
                values={"Field": "Error", "Value": "Extraction failed"},
                confidence=0.0,
                extraction_method='error',
                extraction_success=False,
                missing_fields=[],
                extraction_passes=0
            )
    
    async def generate_summary(self, text: str) -> str:
        """Generate AI summary - UNCHANGED"""
        try:
            prompt = f"""Summarize this document in 2-3 sentences:

{text[:1000]}

Summary:"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=30.0
            )
            
            summary = str(response.get("response", "No summary available") if isinstance(response, dict) else response)
            return summary.strip()[:500]
        except:
            return "Summary unavailable"
    
    async def extract_entities_legacy(self, text: str) -> Dict[str, List[str]]:
        """Legacy entity extraction for backward compatibility - UNCHANGED"""
        try:
            prompt = f"""Extract entities from this text. Return in JSON format:

{text[:1000]}

JSON format:
{{
  "people": ["name1", "name2"],
  "organizations": ["org1"],
  "dates": ["date1"],
  "amounts": ["amount1"]
}}"""

            response = await asyncio.wait_for(
                self.ollama.generate(prompt),
                timeout=45.0
            )
            
            result = str(response.get("response", "{}") if isinstance(response, dict) else response)
            
            try:
                entities = json.loads(result)
                return entities if isinstance(entities, dict) else {}
            except:
                return {"extracted": ["See Excel report for details"]}
        except:
            return {}
# ============================================================================
# ULTIMATE HYBRID OCR ENGINE (UNCHANGED - CORE FUNCTIONALITY)
# ============================================================================
class UltimateHybridOCREngine:
    """Ultimate 3-engine OCR + AI Ollama"""
    
    def __init__(self, ollama_client: Optional[OptimizedOllamaClient] = None):
        self.preprocessor = AdvancedImagePreprocessor()
        self.voting_system = UltimateVotingSystem()
        self.converter = UniversalFileConverter()
        
        # Initialize OCR engines
        self.easyocr_reader = None
        self.paddleocr_engine = None
        
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR initialized")
            except:
                logger.warning("⚠️  EasyOCR initialization failed")
        
        if PADDLEOCR_AVAILABLE and PaddleOCREngine:
            try:
                self.paddleocr_engine = PaddleOCREngine()
                logger.info("✅ PaddleOCR initialized")
            except:
                logger.warning("⚠️  PaddleOCR initialization failed")
        
        # AI Processor
        self.ai_processor = None
        if ollama_client and OLLAMA_AVAILABLE:
            self.ai_processor = EnhancedOllamaAIProcessor(ollama_client)
            logger.info("✅ Enhanced Ollama AI Processor initialized")
    
    def _easyocr_extract(self, image_path: Path) -> OCREngineResult:
        """EasyOCR extraction - UNCHANGED"""
        start = time.time()
        try:
            if not self.easyocr_reader:
                return OCREngineResult("easyocr", "", 0.0, 0.0, False, "Not initialized")
            
            result = self.easyocr_reader.readtext(str(image_path))
            text = "\n".join([detection[1] for detection in result])
            avg_conf = sum([detection[2] for detection in result]) / len(result) if result else 0.0
            
            return OCREngineResult(
                engine="easyocr",
                text=text,
                confidence=avg_conf,
                processing_time=time.time()-start,
                success=True,
                lines_detected=len(result)
            )
        except Exception as e:
            return OCREngineResult("easyocr", "", 0.0, time.time()-start, False, str(e))
    
    def _tesseract_extract(self, image_path: Path) -> OCREngineResult:
        """Tesseract extraction - UNCHANGED"""
        start = time.time()
        try:
            if not TESSERACT_AVAILABLE or not PIL_AVAILABLE:
                return OCREngineResult("tesseract", "", 0.0, 0.0, False, "Not available")
            
            preprocessed = self.preprocessor.preprocess(image_path)
            if preprocessed is not None:
                img = Image.fromarray(preprocessed)
            else:
                img = Image.open(image_path)
            
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, config=custom_config)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            return OCREngineResult(
                engine="tesseract",
                text=text,
                confidence=avg_conf,
                processing_time=time.time()-start,
                success=True,
                lines_detected=len([t for t in data['text'] if t.strip()])
            )
        except Exception as e:
            return OCREngineResult("tesseract", "", 0.0, time.time()-start, False, str(e))
    
    def _paddleocr_extract(self, image_path: Path) -> OCREngineResult:
        """PaddleOCR extraction - UNCHANGED"""
        start = time.time()
        try:
            if not self.paddleocr_engine:
                return OCREngineResult("paddleocr", "", 0.0, 0.0, False, "Not initialized")
            
            result = self.paddleocr_engine.extract_text(str(image_path))
            return OCREngineResult(
                engine="paddleocr",
                text=result.get('text', ''),
                confidence=result.get('confidence', 0.0),
                processing_time=time.time()-start,
                success=True,
                lines_detected=result.get('lines_detected', 0)
            )
        except Exception as e:
            return OCREngineResult("paddleocr", "", 0.0, time.time()-start, False, str(e))
    
    async def process_image(self, image_path: Path) -> ProcessingResult:
        """Complete image processing pipeline - ENHANCED WITH MULTIPASS EXTRACTION"""
        job_id = str(uuid.uuid4())[:8]
        overall_start = time.time()
        
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 PROCESSING JOB {job_id}")
            logger.info(f"{'='*80}")
            
            # Step 1: Convert image
            converted_path = self.converter.convert_to_png(image_path)
            
            # Step 2: Run 3 OCR engines in parallel
            logger.info("\n🔍 RUNNING 3 OCR ENGINES IN PARALLEL")
            loop = asyncio.get_event_loop()
            
            easy_task = loop.run_in_executor(EXECUTOR, self._easyocr_extract, converted_path)
            tess_task = loop.run_in_executor(EXECUTOR, self._tesseract_extract, converted_path)
            paddle_task = loop.run_in_executor(EXECUTOR, self._paddleocr_extract, converted_path)
            
            easy_result, tess_result, paddle_result = await asyncio.gather(easy_task, tess_task, paddle_task)
            
            # Step 3: Voting
            voting_result = self.voting_system.vote(easy_result, tess_result, paddle_result)
            
            if not voting_result.final_text or len(voting_result.final_text.strip()) < 10:
                return ProcessingResult(
                    job_id=job_id,
                    status="error",
                    ocr_text="",
                    cleaned_text="",
                    ocr_confidence=0.0,
                    quality_score=0.0,
                    ocr_engines=[],
                    best_engine="none",
                    summary="OCR failed",
                    entities={},
                    excel_structure=None,
                    processing_time=time.time()-overall_start,
                    error="No text extracted"
                )
            
            # Step 4: AI Processing
            cleaned_text = voting_result.final_text
            summary = "No AI processing available"
            entities = {}
            excel_structure = None
            
            if self.ai_processor:
                logger.info("\n🤖 AI PROCESSING WITH OLLAMA")
                
                # Clean OCR text
                cleaned_result = await self.ai_processor.clean_ocr_text(voting_result.final_text)
                cleaned_text = cleaned_result.cleaned_text
                
                # Generate summary
                summary = await self.ai_processor.generate_summary(cleaned_text)
                
                # Extract entities (legacy format)
                entities = await self.ai_processor.extract_entities_legacy(cleaned_text)
                
                # 🆕 BUILD ENHANCED EXCEL STRUCTURE (MULTI-PASS EXTRACTION)
                excel_structure = await self.ai_processor.build_smart_excel_structure_enhanced(cleaned_text)
            
            # Step 5: Create result
            result = ProcessingResult(
                job_id=job_id,
                status="success",
                ocr_text=voting_result.final_text,
                cleaned_text=cleaned_text,
                ocr_confidence=voting_result.engine_confidence,
                quality_score=voting_result.quality_score,
                ocr_engines=voting_result.engines_used,
                best_engine=voting_result.best_engine,
                summary=summary,
                entities=entities,
                excel_structure=excel_structure,
                processing_time=time.time()-overall_start
            )
            
            logger.info(f"\n✅ JOB {job_id} COMPLETED in {result.processing_time:.2f}s")
            logger.info(f"{'='*80}\n")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            logger.error(traceback.format_exc())
            
            return ProcessingResult(
                job_id=job_id,
                status="error",
                ocr_text="",
                cleaned_text="",
                ocr_confidence=0.0,
                quality_score=0.0,
                ocr_engines=[],
                best_engine="error",
                summary="",
                entities={},
                excel_structure=None,
                processing_time=time.time()-overall_start,
                error=str(e)
            )

# ============================================================================
# PROFESSIONAL EXCEL REPORT GENERATOR (ENHANCED)
# ============================================================================
class ProfessionalExcelReportGenerator:
    """Generate beautiful Excel reports with enhanced data"""
    
    def __init__(self):
        self.styles = self._init_styles()
    
    def _init_styles(self):
        """Initialize Excel styles"""
        if not OPENPYXL_AVAILABLE:
            return {}
        
        return {
            'header': {
                'font': Font(name='Calibri', size=12, bold=True, color='FFFFFF'),
                'fill': PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid'),
                'alignment': Alignment(horizontal='center', vertical='center', wrap_text=True),
                'border': Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            },
            'data': {
                'font': Font(name='Calibri', size=11),
                'alignment': Alignment(horizontal='left', vertical='center', wrap_text=True),
                'border': Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            },
            'title': {
                'font': Font(name='Calibri', size=16, bold=True, color='1F4E78'),
                'alignment': Alignment(horizontal='center', vertical='center')
            }
        }
    
    def generate(self, result: ProcessingResult, output_path: Path) -> bool:
        """Generate Excel report - ENHANCED"""
        try:
            if not OPENPYXL_AVAILABLE:
                logger.warning("⚠️  OpenPyXL not available - using CSV fallback")
                return self._generate_csv_fallback(result, output_path)
            
            logger.info("\n📊 GENERATING PROFESSIONAL EXCEL REPORT")
            
            from openpyxl import Workbook
            wb = Workbook()
            
            # Remove default sheet
            if 'Sheet' in wb.sheetnames:
                del wb['Sheet']
            
            # Sheet 1: Main Data
            self._create_main_data_sheet(wb, result)
            
            # Sheet 2: Raw OCR Text
            self._create_raw_ocr_sheet(wb, result)
            
            # Sheet 3: Processing Details
            self._create_processing_details_sheet(wb, result)
            
            # Save
            wb.save(output_path)
            logger.info(f"✅ Excel report saved: {output_path.name}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Excel generation error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _create_main_data_sheet(self, wb, result: ProcessingResult):
        """Create main data sheet with extracted entities"""
        ws = wb.create_sheet("Extracted Data", 0)
        
        # Title
        ws['A1'] = f"OCR Elite v15.1 - {result.excel_structure.document_type if result.excel_structure else 'Document'} Report"
        ws['A1'].font = self.styles['title']['font']
        ws['A1'].alignment = self.styles['title']['alignment']
        ws.merge_cells('A1:H1')
        ws.row_dimensions[1].height = 30
        
        # Empty row
        ws.append([])
        
        # Headers
        if result.excel_structure and result.excel_structure.columns:
            headers = result.excel_structure.columns
            ws.append(headers)
            
            # Apply header styling
            for col_idx, header in enumerate(headers, start=1):
                cell = ws.cell(row=3, column=col_idx)
                cell.font = self.styles['header']['font']
                cell.fill = self.styles['header']['fill']
                cell.alignment = self.styles['header']['alignment']
                cell.border = self.styles['header']['border']
                
                # Auto-width
                ws.column_dimensions[get_column_letter(col_idx)].width = max(15, len(str(header)) + 2)
            
            # Data row
            values = result.excel_structure.values
            row_data = [values.get(col, "Not found (no unique identifier is provided)") for col in headers]
            ws.append(row_data)
            
            # Apply data styling
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=4, column=col_idx)
                cell.font = self.styles['data']['font']
                cell.alignment = self.styles['data']['alignment']
                cell.border = self.styles['data']['border']
            
            # Add metadata
            ws.append([])
            ws.append(['Metadata:', ''])
            ws.append(['Extraction Method:', result.excel_structure.extraction_method])
            ws.append(['Extraction Passes:', result.excel_structure.extraction_passes])
            ws.append(['Confidence:', f"{result.excel_structure.confidence:.2%}"])
            ws.append(['OCR Engine:', result.best_engine])
            ws.append(['Processing Time:', f"{result.processing_time:.2f}s"])
            
            if result.excel_structure.missing_fields:
                ws.append([])
                ws.append(['Missing Fields:', ', '.join(result.excel_structure.missing_fields)])
        else:
            ws.append(['No data extracted'])
        
        # Freeze panes
        ws.freeze_panes = 'A4'
    
    def _create_raw_ocr_sheet(self, wb, result: ProcessingResult):
        """Create raw OCR text sheet"""
        ws = wb.create_sheet("Raw OCR Text", 1)
        
        ws['A1'] = "Raw OCR Text"
        ws['A1'].font = self.styles['title']['font']
        ws.merge_cells('A1:D1')
        
        ws.append([])
        ws.append(['OCR Text:'])
        
        # Split text into lines
        lines = result.ocr_text.split('\n')
        for line in lines[:500]:  # Limit to 500 lines
            ws.append([line])
        
        ws.column_dimensions['A'].width = 100
    
    def _create_processing_details_sheet(self, wb, result: ProcessingResult):
        """Create processing details sheet"""
        ws = wb.create_sheet("Processing Details", 2)
        
        ws['A1'] = "Processing Details"
        ws['A1'].font = self.styles['title']['font']
        ws.merge_cells('A1:B1')
        
        ws.append([])
        ws.append(['Job ID:', result.job_id])
        ws.append(['Status:', result.status])
        ws.append(['OCR Confidence:', f"{result.ocr_confidence:.2%}"])
        ws.append(['Quality Score:', f"{result.quality_score:.2%}"])
        ws.append(['Best Engine:', result.best_engine])
        ws.append(['Engines Used:', ', '.join(result.ocr_engines)])
        ws.append(['Processing Time:', f"{result.processing_time:.2f}s"])
        ws.append([])
        ws.append(['Summary:'])
        ws.append([result.summary])
        
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 50
    
    def _generate_csv_fallback(self, result: ProcessingResult, output_path: Path) -> bool:
        """CSV fallback when Excel not available"""
        try:
            csv_path = output_path.with_suffix('.csv')
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                if result.excel_structure and result.excel_structure.columns:
                    writer.writerow(result.excel_structure.columns)
                    values = result.excel_structure.values
                    row_data = [values.get(col, "Not found") for col in result.excel_structure.columns]
                    writer.writerow(row_data)
                else:
                    writer.writerow(['Field', 'Value'])
                    writer.writerow(['No data', 'extracted'])
            
            logger.info(f"✅ CSV report saved: {csv_path.name}")
            return True
        except:
            return False

# ============================================================================
# JSON REPORT GENERATOR (UNCHANGED)
# ============================================================================
class JSONReportGenerator:
    """Generate JSON reports"""
    
    @staticmethod
    def generate(result: ProcessingResult, output_path: Path) -> bool:
        try:
            report = {
                'job_id': result.job_id,
                'status': result.status,
                'document_type': result.excel_structure.document_type if result.excel_structure else 'Unknown',
                'ocr_confidence': result.ocr_confidence,
                'quality_score': result.quality_score,
                'best_engine': result.best_engine,
                'engines_used': result.ocr_engines,
                'processing_time': result.processing_time,
                'extracted_data': result.excel_structure.values if result.excel_structure else {},
                'summary': result.summary,
                'raw_ocr_text': result.ocr_text[:1000],
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ JSON report saved: {output_path.name}")
            return True
        except:
            return False
# ============================================================================
# GLOBAL INSTANCES
# ============================================================================
ocr_engine: Optional[UltimateHybridOCREngine] = None
excel_generator: Optional[ProfessionalExcelReportGenerator] = None
json_generator: Optional[JSONReportGenerator] = None

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global ocr_engine, excel_generator, json_generator
    
    logger.info("\n" + "="*80)
    logger.info("🚀 OCR ELITE v15.1 ENHANCED EXTRACTION - STARTING UP")
    logger.info("="*80 + "\n")
    
    # Initialize Ollama client
    ollama_client = None
    if OLLAMA_AVAILABLE and OptimizedOllamaClient:
        try:
            config = OllamaConfig(
                base_url="http://localhost:11434",
                model="llama3.1:8b",
                timeout=120,
                max_retries=3
            )
            ollama_client = OptimizedOllamaClient(config)
            logger.info("✅ Ollama client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Ollama initialization failed: {e}")
    
    # Initialize OCR engine
    ocr_engine = UltimateHybridOCREngine(ollama_client)
    logger.info("✅ Ultimate Hybrid OCR Engine initialized")
    
    # Initialize report generators
    excel_generator = ProfessionalExcelReportGenerator()
    json_generator = JSONReportGenerator()
    logger.info("✅ Report generators initialized")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL SYSTEMS READY - OCR ELITE v15.1 ONLINE")
    logger.info("="*80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("\n🛑 Shutting down OCR Elite v15.1...")
    EXECUTOR.shutdown(wait=True)
    logger.info("✅ Shutdown complete\n")

# ============================================================================
# FASTAPI ROUTES - ALL OLD ROUTES PRESERVED + NEW ROUTES ADDED
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - UNCHANGED"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.warning(f"Template not found: {e}")
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>OCR Elite v15.1</title></head>
<body style="font-family: Arial; text-align: center; padding: 50px;">
    <h1>🚀 OCR Elite v15.1 Enhanced</h1>
    <p>Multi-Pass Entity Extraction System</p>
    <p><a href="/ocr">Go to OCR</a> | <a href="/ner">Go to NER</a></p>
    <p><a href="/docs">API Documentation</a></p>
</body>
</html>
        """)

@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page(request: Request):
    """OCR page - UNCHANGED"""
    try:
        return templates.TemplateResponse("ocr.html", {"request": request})
    except Exception as e:
        logger.warning(f"Template not found: {e}")
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>OCR - Upload</title></head>
<body style="font-family: Arial; padding: 50px;">
    <h1>📄 OCR Upload</h1>
    <form action="/ocr/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Process Document</button>
    </form>
    <p><a href="/">← Back to Home</a></p>
</body>
</html>
        """)

@app.post("/ocr/upload")
async def ocr_upload(request: Request, file: UploadFile = File(...)):
    """OCR upload endpoint - UNCHANGED ROUTE, ENHANCED PROCESSING"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")
        
        # Save file
        filepath = UPLOAD_DIR / file.filename
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        logger.info(f"📥 Uploaded: {file.filename}")
        
        # Process with enhanced multi-pass extraction
        if not ocr_engine:
            raise HTTPException(status_code=500, detail="OCR engine not initialized")
        
        result = await ocr_engine.process_image(filepath)
        
        if result.status == "success":
            # Generate reports
            if excel_generator:
                excel_filename = f"{result.job_id}_report.xlsx"
                excel_path = OUTPUT_DIR / excel_filename
                excel_generator.generate(result, excel_path)
                result.excel_report = str(excel_path)
            
            if json_generator:
                json_filename = f"{result.job_id}_report.json"
                json_path = OUTPUT_DIR / json_filename
                json_generator.generate(result, json_path)
                result.json_report = str(json_path)
            
            # Try to render template
            try:
                return templates.TemplateResponse("ocr_result.html", {
                    "request": request,
                    "job_id": result.job_id,
                    "ocr_text": result.ocr_text,
                    "cleaned_text": result.cleaned_text,
                    "summary": result.summary,
                    "entities": result.entities,
                    "confidence": result.ocr_confidence,
                    "engines_used": result.ocr_engines,
                    "best_engine": result.best_engine,
                    "quality_score": result.quality_score,
                    "processing_time": result.processing_time,
                    "excel_report": result.excel_report,
                    "json_report": result.json_report,
                    "document_type": result.excel_structure.document_type if result.excel_structure else "Document",
                    "excel_columns": result.excel_structure.columns if result.excel_structure else [],
                    "extraction_method": result.excel_structure.extraction_method if result.excel_structure else "unknown"
                })
            except Exception as template_error:
                logger.warning(f"Template error: {template_error}")
                # Fallback to JSON response
                return JSONResponse(content={
                    "job_id": result.job_id,
                    "status": result.status,
                    "ocr_text": result.ocr_text[:1000],
                    "confidence": result.ocr_confidence,
                    "quality_score": result.quality_score,
                    "best_engine": result.best_engine,
                    "document_type": result.excel_structure.document_type if result.excel_structure else "Document",
                    "extraction_method": result.excel_structure.extraction_method if result.excel_structure else "unknown",
                    "download_excel": f"/download/{result.job_id}/excel",
                    "download_json": f"/download/{result.job_id}/json"
                })
        else:
            raise HTTPException(status_code=500, detail=result.error or "Processing failed")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ner", response_class=HTMLResponse)
async def ner_page(request: Request):
    """NER page - UNCHANGED"""
    try:
        return templates.TemplateResponse("ner.html", {"request": request})
    except Exception as e:
        logger.warning(f"Template not found: {e}")
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>NER - Entity Extraction</title></head>
<body style="font-family: Arial; padding: 50px;">
    <h1>🔍 Named Entity Recognition</h1>
    <form action="/ner/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Extract Entities</button>
    </form>
    <p><a href="/">← Back to Home</a></p>
</body>
</html>
        """)

@app.post("/ner/upload")
async def ner_upload(request: Request, file: UploadFile = File(...)):
    """NER upload endpoint - UNCHANGED"""
    try:
        # Save file
        filepath = UPLOAD_DIR / file.filename
        with open(filepath, 'wb') as f:
            f.write(await file.read())
        
        # Process
        if not ocr_engine:
            raise HTTPException(status_code=500, detail="OCR engine not initialized")
        
        result = await ocr_engine.process_image(filepath)
        
        if result.status == "success":
            try:
                return templates.TemplateResponse("ner_result.html", {
                    "request": request,
                    "job_id": result.job_id,
                    "entities": result.entities,
                    "summary": result.summary,
                    "confidence": result.quality_score
                })
            except:
                return JSONResponse(content={
                    "job_id": result.job_id,
                    "entities": result.entities,
                    "summary": result.summary,
                    "confidence": result.quality_score
                })
        else:
            raise HTTPException(status_code=500, detail=result.error)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NER upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/success", response_class=HTMLResponse)
async def success_page(request: Request):
    """Success page - UNCHANGED"""
    try:
        return templates.TemplateResponse("success.html", {"request": request})
    except:
        return HTMLResponse("<h1>✅ Success</h1><p><a href='/'>Back to Home</a></p>")

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Main upload endpoint - UNCHANGED"""
    try:
        if not file.filename:
            raise HTTPException(400, "No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(400, f"Unsupported format: {file_ext}")
        
        filepath = UPLOAD_DIR / file.filename
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, "File too large")
        
        with open(filepath, 'wb') as f:
            f.write(content)
        
        if not ocr_engine:
            raise HTTPException(500, "OCR engine not initialized")
        
        result = await ocr_engine.process_image(filepath)
        
        # Convert to dict for JSON response
        result_dict = asdict(result)
        return JSONResponse(content=result_dict)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/process")
async def api_process(file: UploadFile = File(...), use_voting: bool = True):
    """API processing endpoint - UNCHANGED"""
    try:
        filepath = UPLOAD_DIR / file.filename
        with open(filepath, 'wb') as f:
            f.write(await file.read())
        
        if not ocr_engine:
            raise HTTPException(500, "OCR engine not initialized")
        
        result = await ocr_engine.process_image(filepath)
        return JSONResponse(content=asdict(result))
    
    except Exception as e:
        logger.error(f"API process error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/batch-process")
async def batch_process(files: List[UploadFile] = File(...), use_voting: bool = True):
    """Batch processing endpoint - UNCHANGED"""
    try:
        results = []
        for file in files:
            try:
                filepath = UPLOAD_DIR / file.filename
                with open(filepath, 'wb') as f:
                    f.write(await file.read())
                
                if not ocr_engine:
                    raise HTTPException(500, "OCR engine not initialized")
                
                result = await ocr_engine.process_image(filepath)
                results.append({
                    "filename": file.filename,
                    "job_id": result.job_id,
                    "status": result.status,
                    "quality_score": result.quality_score,
                    "document_type": result.excel_structure.document_type if result.excel_structure else "Unknown",
                    "extraction_method": result.excel_structure.extraction_method if result.excel_structure else "unknown"
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return JSONResponse(content={"results": results, "total": len(files)})
    
    except Exception as e:
        logger.error(f"Batch process error: {e}")
        raise HTTPException(500, str(e))

# ============================================================================
# DOWNLOAD ROUTES - OLD FORMAT (MUST HAVE THIS!)
# ============================================================================

@app.get("/download/{jobid}/{filetype}")
async def download_report_old(jobid: str, filetype: str):
    """Download reports - OLD ROUTE FORMAT (UNCHANGED) - THIS FIXES YOUR 404!"""
    try:
        if filetype not in ['excel', 'json']:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Construct filename
        extension = '.xlsx' if filetype == 'excel' else '.json'
        filepath = OUTPUT_DIR / f"{jobid}_report{extension}"
        
        logger.info(f"📥 Download request: {filepath}")
        
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            raise HTTPException(status_code=404, detail=f"Report not found: {filepath.name}")
        
        mediatype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if filetype == 'excel' else "application/json"
        
        return FileResponse(
            path=filepath,
            filename=filepath.name,
            media_type=mediatype
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEW API V1 ROUTES
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "15.1.0",
        "features": {
            "easyocr": EASYOCR_AVAILABLE,
            "tesseract": TESSERACT_AVAILABLE,
            "paddleocr": PADDLEOCR_AVAILABLE,
            "ollama": OLLAMA_AVAILABLE,
            "openpyxl": OPENPYXL_AVAILABLE,
        },
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/api/v1/process")
async def process_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process uploaded document (image) with OCR + AI (NEW ENHANCED ENDPOINT)
    """
    upload_start = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Read file
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Save uploaded file
        job_id = str(uuid.uuid4())[:8]
        upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        
        with open(upload_path, 'wb') as f:
            f.write(contents)
        
        logger.info(f"\n📥 File uploaded: {file.filename} ({len(contents)} bytes)")
        
        # Process with OCR engine
        if not ocr_engine:
            raise HTTPException(status_code=500, detail="OCR engine not initialized")
        
        result = await ocr_engine.process_image(upload_path)
        
        # Generate reports
        if result.status == "success":
            # Excel report
            excel_filename = f"{job_id}_report.xlsx"
            excel_path = OUTPUT_DIR / excel_filename
            
            if excel_generator:
                excel_success = excel_generator.generate(result, excel_path)
                if excel_success:
                    result.excel_report = str(excel_path)
            
            # JSON report
            json_filename = f"{job_id}_report.json"
            json_path = OUTPUT_DIR / json_filename
            
            if json_generator:
                json_success = json_generator.generate(result, json_path)
                if json_success:
                    result.json_report = str(json_path)
        
        # Build response
        response_data = {
            "job_id": result.job_id,
            "status": result.status,
            "filename": file.filename,
            "processing_time": result.processing_time,
            "upload_time": time.time() - upload_start,
            "ocr": {
                "text": result.ocr_text[:2000],
                "cleaned_text": result.cleaned_text[:2000],
                "confidence": result.ocr_confidence,
                "quality_score": result.quality_score,
                "best_engine": result.best_engine,
                "engines_used": result.ocr_engines
            },
            "ai": {
                "summary": result.summary,
                "entities": result.entities
            },
            "excel": None,
            "reports": {
                "excel": excel_filename if result.excel_report else None,
                "json": json_filename if result.json_report else None
            },
            "error": result.error
        }
        
        # Add Excel structure details
        if result.excel_structure:
            response_data["excel"] = {
                "document_type": result.excel_structure.document_type,
                "columns": result.excel_structure.columns,
                "values": result.excel_structure.values,
                "confidence": result.excel_structure.confidence,
                "extraction_method": result.excel_structure.extraction_method,
                "extraction_passes": result.excel_structure.extraction_passes,
                "missing_fields": result.excel_structure.missing_fields,
                "extraction_success": result.excel_structure.extraction_success
            }
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, upload_path)
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/download/{filename}")
async def download_report(filename: str):
    """Download generated report"""
    try:
        file_path = OUTPUT_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        if filename.endswith('.xlsx'):
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.endswith('.json'):
            media_type = "application/json"
        elif filename.endswith('.csv'):
            media_type = "text/csv"
        else:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/reports")
async def list_reports():
    """List all available reports"""
    try:
        reports = []
        
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file():
                reports.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created": datetime.datetime.fromtimestamp(
                        file_path.stat().st_ctime
                    ).isoformat(),
                    "type": file_path.suffix[1:]
                })
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created"], reverse=True)
        
        return {"reports": reports, "total": len(reports)}
    
    except Exception as e:
        logger.error(f"❌ List reports error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/reports/{filename}")
async def delete_report(filename: str):
    """Delete a report"""
    try:
        file_path = OUTPUT_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        logger.info(f"🗑️  Deleted report: {filename}")
        
        return {"status": "success", "message": f"Deleted {filename}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch")
async def batch_process_v1(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Batch process multiple documents"""
    try:
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
        
        results = []
        
        for file in files:
            try:
                result_response = await process_document(file, background_tasks)
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "result": result_response
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "total": len(files),
            "success": success_count,
            "failed": len(files) - success_count,
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def cleanup_old_files(file_path: Path, max_age_hours: int = 24):
    """Clean up old files"""
    try:
        if file_path.exists():
            age = time.time() - file_path.stat().st_mtime
            if age > max_age_hours * 3600:
                file_path.unlink()
                logger.info(f"🗑️  Cleaned up old file: {file_path.name}")
    except:
        pass

async def cleanup_old_reports():
    """Clean up old reports"""
    try:
        max_age_hours = 48
        now = time.time()
        
        for file_path in OUTPUT_DIR.iterdir():
            if file_path.is_file():
                age = now - file_path.stat().st_mtime
                if age > max_age_hours * 3600:
                    file_path.unlink()
                    logger.info(f"🗑️  Cleaned up old report: {file_path.name}")
        
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                age = now - file_path.stat().st_mtime
                if age > max_age_hours * 3600:
                    file_path.unlink()
                    logger.info(f"🗑️  Cleaned up old upload: {file_path.name}")
    except:
        pass

# ============================================================================
# HTML TEMPLATE CREATION (AUTO-GENERATE IF NOT EXISTS)
# ============================================================================
def create_default_templates():
    """Create default HTML templates if they don't exist"""
    
    # index.html
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Elite v15.1</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        h1 { color: #667eea; text-align: center; margin-bottom: 20px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 40px; }
        .button { display: inline-block; padding: 15px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 30px; margin: 10px; transition: transform 0.3s; }
        .button:hover { transform: translateY(-2px); }
        .links { text-align: center; margin-top: 40px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 OCR Elite v15.1</h1>
        <p class="subtitle">Enhanced Multi-Pass Entity Extraction System</p>
        <div class="links">
            <a href="/ocr" class="button">📄 OCR Processing</a>
            <a href="/ner" class="button">🔍 Entity Extraction</a>
            <a href="/docs" class="button">📚 API Docs</a>
        </div>
    </div>
</body>
</html>""")
    
    # ocr.html
    ocr_path = TEMPLATES_DIR / "ocr.html"
    if not ocr_path.exists():
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>OCR Upload</title>
    <style>
        body { font-family: Arial; padding: 50px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
        h1 { color: #667eea; }
        form { margin-top: 30px; }
        input[type="file"] { margin: 20px 0; }
        button { padding: 15px 30px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #5568d3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 OCR Document Processing</h1>
        <p>Upload an image to extract text with multi-pass AI extraction</p>
        <form action="/ocr/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br>
            <button type="submit">Process Document</button>
        </form>
        <p><a href="/">← Back to Home</a></p>
    </div>
</body>
</html>""")
    
    # ner.html
    ner_path = TEMPLATES_DIR / "ner.html"
    if not ner_path.exists():
        with open(ner_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>NER Upload</title>
    <style>
        body { font-family: Arial; padding: 50px; background: #f5f5f5; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
        h1 { color: #667eea; }
        form { margin-top: 30px; }
        input[type="file"] { margin: 20px 0; }
        button { padding: 15px 30px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #5568d3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Named Entity Recognition</h1>
        <p>Upload an image to extract entities</p>
        <form action="/ner/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br>
            <button type="submit">Extract Entities</button>
        </form>
        <p><a href="/">← Back to Home</a></p>
    </div>
</body>
</html>""")
    
    logger.info("✅ Default templates created")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Create default templates
    create_default_templates()
    
    # Run server
    logger.info("\n" + "="*80)
    logger.info("🚀 STARTING OCR ELITE v15.1 ENHANCED EXTRACTION SERVER")
    logger.info("="*80)
    logger.info("📍 Server URL: http://localhost:8000")
    logger.info("📍 OCR Page: http://localhost:8000/ocr")
    logger.info("📍 NER Page: http://localhost:8000/ner")
    logger.info("📍 API Docs: http://localhost:8000/docs")
    logger.info("📍 Health Check: http://localhost:8000/health")
    logger.info("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
