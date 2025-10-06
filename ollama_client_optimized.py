#!/usr/bin/env python
"""
ollama_client_optimized.py - ULTRA-FAST OLLAMA CLIENT WITH ASYNC OPTIMIZATION

Performance Enhancements:
- Async HTTP with Connection Pooling
- Request Batching & Parallelization
- Streaming Response Handling
- Intelligent Retry Logic
- Response Caching
- Token Prediction Optimization
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
from diskcache import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Response cache
CACHE_DIR = Path("./cache/ollama")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
response_cache = Cache(str(CACHE_DIR))


@dataclass
class OllamaConfig:
    """Ollama configuration with optimized defaults"""
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    timeout: int = 120  # Reduced from 300s
    max_retries: int = 2  # Reduced from 3
    temperature: float = 0.0  # Deterministic
    num_predict: int = 8192  # Max tokens for Llama 3.1
    repeat_penalty: float = 1.2  # Reduce repetition
    top_k: int = 20
    top_p: float = 0.85
    num_ctx: int = 8192  # Context window
    enable_caching: bool = True
    stream: bool = False  # Set True for real-time responses


class OptimizedOllamaClient:
    """
    High-performance async Ollama client with connection pooling
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._connector = None
        
        logger.info(f"Initialized Ollama Client (Model: {self.config.model})")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _create_session(self):
        """Create optimized aiohttp session with connection pooling"""
        if self.session is None or self.session.closed:
            # Configure connection pooling for better performance
            self._connector = aiohttp.TCPConnector(
                limit=100,  # Max 100 concurrent connections
                limit_per_host=50,  # Max 50 per Ollama host
                ttl_dns_cache=300,  # Cache DNS for 5 min
                keepalive_timeout=60  # Keep connections alive
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=10,
                sock_read=self.config.timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
            
            logger.info("✓ Async HTTP session created with connection pooling")
    
    async def _close_session(self):
        """Close session and cleanup"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("✓ Async HTTP session closed")
    
    def _compute_prompt_hash(self, prompt: str, system_prompt: str = "") -> str:
        """Compute hash for caching"""
        combined = f"{system_prompt}|||{prompt}|||{self.config.model}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        format_json: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate completion with optimized async request
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            format_json: Force JSON output
            use_cache: Use response caching
            
        Returns:
            {
                'response': str,
                'processing_time': float,
                'from_cache': bool,
                'tokens': int
            }
        """
        start_time = time.time()
        
        # Check cache
        if use_cache and self.config.enable_caching:
            cache_key = self._compute_prompt_hash(prompt, system_prompt or "")
            cached = response_cache.get(cache_key)
            if cached:
                cached['from_cache'] = True
                logger.info(f"✓ Cache hit for prompt {cache_key[:8]}")
                return cached
        
        # Ensure session exists
        if self.session is None or self.session.closed:
            await self._create_session()
        
        # Build request payload
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": self.config.stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "num_ctx": self.config.num_ctx
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if format_json:
            payload["format"] = "json"
        
        # Execute request with retry logic
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
                    if self.config.stream:
                        # Handle streaming response
                        full_response = ""
                        async for line in response.content:
                            if line:
                                try:
                                    chunk = json.loads(line.decode('utf-8'))
                                    if 'response' in chunk:
                                        full_response += chunk['response']
                                except json.JSONDecodeError:
                                    continue
                        result_text = full_response
                    else:
                        # Handle complete response
                        data = await response.json()
                        result_text = data.get('response', '')
                    
                    processing_time = time.time() - start_time
                    
                    result = {
                        'response': result_text.strip(),
                        'processing_time': processing_time,
                        'from_cache': False,
                        'tokens': len(result_text.split())  # Approximate
                    }
                    
                    # Cache successful result
                    if use_cache and self.config.enable_caching:
                        response_cache.set(cache_key, result, expire=3600)  # 1h cache
                    
                    logger.info(f"✓ Generated response ({result['tokens']} tokens) in {processing_time:.2f}s")
                    
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.config.max_retries}")
                if attempt == self.config.max_retries - 1:
                    raise Exception("Ollama request timed out after retries")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Ollama request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Ollama request failed after all retries")
    
    async def batch_generate(
        self,
        prompts: List[Dict[str, str]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts concurrently with controlled parallelism
        
        Args:
            prompts: List of {'prompt': str, 'system_prompt': str (optional)}
            max_concurrent: Max parallel requests (default: 5)
            
        Returns:
            List of response dicts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _generate_with_semaphore(prompt_dict):
            async with semaphore:
                return await self.generate(
                    prompt=prompt_dict['prompt'],
                    system_prompt=prompt_dict.get('system_prompt'),
                    format_json=prompt_dict.get('format_json', False)
                )
        
        tasks = [_generate_with_semaphore(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {idx} failed: {result}")
                processed_results.append({
                    'response': '',
                    'error': str(result),
                    'processing_time': 0.0,
                    'from_cache': False
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def health_check(self) -> bool:
        """Check if Ollama server is healthy"""
        try:
            if self.session is None or self.session.closed:
                await self._create_session()
            
            async with self.session.get(f"{self.config.base_url}/api/tags") as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Singleton instance
_ollama_client = None


async def get_ollama_client(config: Optional[OllamaConfig] = None) -> OptimizedOllamaClient:
    """Get or create optimized Ollama client singleton"""
    global _ollama_client
    
    if _ollama_client is None:
        _ollama_client = OptimizedOllamaClient(config)
        await _ollama_client._create_session()
    
    return _ollama_client
