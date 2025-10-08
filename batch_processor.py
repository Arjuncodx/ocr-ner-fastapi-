#!/usr/bin/env python
"""
batch_processor.py - Fast Batch OCR Processing with Templates
=============================================================
Processes multiple documents using learned templates.
Integrates with existing OCR Elite v15.2 system.

Senior Python OCR Developer - Fortune 500 Grade
October 2025
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class BatchProcessingResult:
    """Result of batch processing operation"""
    batch_id: str
    template_id: str
    total_documents: int
    successful: int
    failed: int
    processing_time: float
    extracted_data: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# BATCH PROCESSOR
# ============================================================================
class FastBatchProcessor:
    """Fast batch processing using templates (skips AI extraction)"""
    
    def __init__(self, ocr_engine, template_manager):
        """
        Initialize batch processor
        
        Args:
            ocr_engine: Your existing UltimateHybridOCREngine instance
            template_manager: BatchTemplateManager instance
        """
        self.ocr_engine = ocr_engine
        self.template_manager = template_manager
        logger.info("✅ FastBatchProcessor initialized")
    
    async def process_batch_with_template(
        self,
        file_paths: List[Path],
        template_id: str
    ) -> BatchProcessingResult:
        """
        Process multiple documents using a template (FAST - No AI)
        
        Args:
            file_paths: List of image file paths to process
            template_id: ID of approved template to use
            
        Returns:
            BatchProcessingResult with all extracted data
        """
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 BATCH PROCESSING START: {batch_id}")
        logger.info(f"   Template: {template_id}")
        logger.info(f"   Documents: {len(file_paths)}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Load template
            template = self.template_manager.load_template(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            
            if not template.approved:
                raise ValueError(f"Template not approved: {template_id}")
            
            # Process documents
            extracted_data = []
            errors = []
            successful = 0
            failed = 0
            
            for idx, file_path in enumerate(file_paths, 1):
                try:
                    logger.info(f"📄 Processing {idx}/{len(file_paths)}: {file_path.name}")
                    
                    # Run ONLY OCR (skip AI extraction)
                    ocr_text = await self._fast_ocr_only(file_path)
                    
                    if not ocr_text or len(ocr_text.strip()) < 10:
                        raise ValueError("OCR failed or text too short")
                    
                    # Extract values using template (FAST)
                    values = self.template_manager.extract_values_with_template(
                        ocr_text,
                        template
                    )
                    
                    # Add filename for reference
                    values['_filename'] = file_path.name
                    values['_batch_id'] = batch_id
                    
                    extracted_data.append(values)
                    successful += 1
                    
                    logger.info(f"   ✅ Success: {len(values)} fields extracted")
                
                except Exception as e:
                    error_msg = f"{file_path.name}: {str(e)}"
                    errors.append(error_msg)
                    failed += 1
                    logger.error(f"   ❌ Failed: {e}")
            
            processing_time = time.time() - start_time
            
            result = BatchProcessingResult(
                batch_id=batch_id,
                template_id=template_id,
                total_documents=len(file_paths),
                successful=successful,
                failed=failed,
                processing_time=processing_time,
                extracted_data=extracted_data,
                errors=errors
            )
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ BATCH PROCESSING COMPLETE: {batch_id}")
            logger.info(f"   Success: {successful}/{len(file_paths)}")
            logger.info(f"   Failed: {failed}/{len(file_paths)}")
            logger.info(f"   Time: {processing_time:.2f}s")
            logger.info(f"   Avg: {processing_time/len(file_paths):.2f}s per document")
            logger.info(f"{'='*80}\n")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            raise
    
    async def _fast_ocr_only(self, image_path: Path) -> str:
        """
        Fast OCR-only extraction (no AI processing)
        Reuses existing OCR engines without Ollama
        """
        try:
            # Convert image if needed
            converted_path = self.ocr_engine.converter.convert_to_png(image_path)
            
            # Run 3 OCR engines in parallel (reuse existing code)
            loop = asyncio.get_event_loop()
            
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=3)
            
            easy_task = loop.run_in_executor(
                executor,
                self.ocr_engine._easyocr_extract,
                converted_path
            )
            tess_task = loop.run_in_executor(
                executor,
                self.ocr_engine._tesseract_extract,
                converted_path
            )
            paddle_task = loop.run_in_executor(
                executor,
                self.ocr_engine._paddleocr_extract,
                converted_path
            )
            
            easy_result, tess_result, paddle_result = await asyncio.gather(
                easy_task, tess_task, paddle_task
            )
            
            # Vote for best engine (reuse existing voting system)
            voting_result = self.ocr_engine.voting_system.vote(
                easy_result, tess_result, paddle_result
            )
            
            return voting_result.final_text
        
        except Exception as e:
            logger.error(f"❌ Fast OCR error: {e}")
            raise


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
# Initialize in main.py after ocr_engine is created
fast_batch_processor: Optional[FastBatchProcessor] = None
