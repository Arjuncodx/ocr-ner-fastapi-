#!/usr/bin/env python
"""
batch_template_manager.py - Batch Template Learning System
==========================================================
Manages template learning, storage, and reuse for batch processing.
Integrates with existing OCR Elite v15.2 system.

Senior Python OCR Developer - Fortune 500 Grade
October 2025
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class BatchTemplate:
    """Template structure for batch processing"""
    template_id: str
    document_type: str
    columns: List[str]
    sample_values: Dict[str, str]
    created_at: str
    approved: bool = False
    total_documents_processed: int = 0


# ============================================================================
# TEMPLATE MANAGER
# ============================================================================
class BatchTemplateManager:
    """Manages batch processing templates - learns from first document, applies to rest"""
    
    def __init__(self, templates_dir: Path = None):
        """Initialize template manager"""
        self.templates_dir = templates_dir or Path("batch_templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ BatchTemplateManager initialized: {self.templates_dir}")
    
    def create_template_from_result(self, result, template_name: str = None) -> BatchTemplate:
        """
        Create template from ProcessingResult (from existing OCR pipeline)
        
        Args:
            result: ProcessingResult object from your existing pipeline
            template_name: Optional custom name for template
            
        Returns:
            BatchTemplate object
        """
        try:
            template_id = str(uuid.uuid4())[:8]
            
            # Extract structure from existing result
            if result.excel_structure and result.excel_structure.columns:
                columns = result.excel_structure.columns
                sample_values = result.excel_structure.values
                document_type = result.excel_structure.document_type
            else:
                raise ValueError("No excel_structure found in result")
            
            template = BatchTemplate(
                template_id=template_id,
                document_type=template_name or document_type,
                columns=columns,
                sample_values=sample_values,
                created_at=datetime.now().isoformat(),
                approved=False,
                total_documents_processed=1
            )
            
            logger.info(f"✅ Template created: {template_id} with {len(columns)} columns")
            return template
        
        except Exception as e:
            logger.error(f"❌ Template creation error: {e}")
            raise
    
    def save_template(self, template: BatchTemplate) -> bool:
        """Save template to disk"""
        try:
            filepath = self.templates_dir / f"{template.template_id}.json"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(template), f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Template saved: {filepath.name}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Template save error: {e}")
            return False
    
    def load_template(self, template_id: str) -> Optional[BatchTemplate]:
        """Load template from disk"""
        try:
            filepath = self.templates_dir / f"{template_id}.json"
            
            if not filepath.exists():
                logger.warning(f"⚠️  Template not found: {template_id}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            template = BatchTemplate(**data)
            logger.info(f"✅ Template loaded: {template_id}")
            return template
        
        except Exception as e:
            logger.error(f"❌ Template load error: {e}")
            return None
    
    def approve_template(self, template_id: str) -> bool:
        """Mark template as approved for batch processing"""
        try:
            template = self.load_template(template_id)
            if not template:
                return False
            
            template.approved = True
            self.save_template(template)
            
            logger.info(f"✅ Template approved: {template_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Template approval error: {e}")
            return False
    
    def list_templates(self) -> List[Dict]:
        """List all available templates"""
        try:
            templates = []
            
            for filepath in self.templates_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    templates.append({
                        'template_id': data.get('template_id'),
                        'document_type': data.get('document_type'),
                        'columns_count': len(data.get('columns', [])),
                        'created_at': data.get('created_at'),
                        'approved': data.get('approved', False),
                        'total_processed': data.get('total_documents_processed', 0)
                    })
                except:
                    continue
            
            templates.sort(key=lambda x: x['created_at'], reverse=True)
            logger.info(f"✅ Found {len(templates)} templates")
            return templates
        
        except Exception as e:
            logger.error(f"❌ List templates error: {e}")
            return []
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template"""
        try:
            filepath = self.templates_dir / f"{template_id}.json"
            
            if not filepath.exists():
                logger.warning(f"⚠️  Template not found: {template_id}")
                return False
            
            filepath.unlink()
            logger.info(f"✅ Template deleted: {template_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Template delete error: {e}")
            return False
    
    def extract_values_with_template(self, ocr_text: str, template: BatchTemplate) -> Dict[str, str]:
        """
        Fast extraction using template (no AI needed)
        Uses simple pattern matching based on field names
        
        Args:
            ocr_text: Raw OCR text from document
            template: BatchTemplate to use
            
        Returns:
            Dict of field_name: value pairs
        """
        try:
            extracted_values = {}
            
            # Split text into lines
            lines = ocr_text.split('\n')
            text_lower = ocr_text.lower()
            
            for field_name in template.columns:
                # Skip Document Type field
                if field_name.lower() == 'document type':
                    extracted_values[field_name] = template.document_type
                    continue
                
                # Try to find field value using simple matching
                field_lower = field_name.lower()
                
                # Look for "field_name: value" pattern
                for line in lines:
                    line_lower = line.lower()
                    
                    # Check if field name is in this line
                    if field_lower in line_lower and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            value = parts[1].strip()
                            if value and len(value) > 0:
                                extracted_values[field_name] = value
                                break
                
                # If not found, leave empty
                if field_name not in extracted_values:
                    extracted_values[field_name] = ""
            
            logger.info(f"✅ Extracted {len(extracted_values)}/{len(template.columns)} values using template")
            return extracted_values
        
        except Exception as e:
            logger.error(f"❌ Template extraction error: {e}")
            return {}


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================
# Can be imported in main.py
batch_template_manager = BatchTemplateManager()
