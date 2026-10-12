"""
CodeInsight AI - CodeBERT Encoder

Provides lazy-loaded CodeBERT embeddings for code snippets.
Supports both CPU and GPU inference. Model is loaded once and reused.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Module-level singleton to avoid reloading
_encoder_instance: Optional["CodeBERTEncoder"] = None


class CodeBERTEncoder:
    """
    Wraps microsoft/codebert-base for generating code embeddings.
    
    Uses mean pooling over non-padding tokens from the last hidden state
    to produce a fixed-size vector representation of code.
    """
    
    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = "auto"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"CodeBERTEncoder initialized (device={self.device}, loaded={self._loaded})")
    
    def load(self) -> bool:
        """
        Lazily load the CodeBERT model and tokenizer.
        Returns True if successful, False otherwise.
        """
        if self._loaded:
            return True
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info(f"Loading CodeBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"CodeBERT loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load CodeBERT: {e}")
            self._loaded = False
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def is_available(self) -> bool:
        """Check if CodeBERT model dependencies are available for loading."""
        if self._loaded:
            return True
        try:
            import transformers
            return True
        except ImportError:
            return False
    
    def encode(self, code: str, max_length: int = 512) -> Optional[np.ndarray]:
        """
        Encode a code snippet into a fixed-size embedding vector.
        
        Uses mean pooling over non-padding tokens from the last hidden state.
        The embedding is L2-normalized.
        
        Args:
            code: Python source code string
            max_length: Maximum token sequence length
            
        Returns:
            Normalized embedding vector or None if encoding fails
        """
        if not self.load():
            return None
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass (no gradient needed for inference)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over non-padding tokens
            # outputs.last_hidden_state shape: (1, seq_len, hidden_size)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            
            # Zero out padding positions
            masked_hidden = last_hidden * attention_mask
            
            # Sum and divide by number of non-padding tokens
            sum_hidden = masked_hidden.sum(dim=1)  # (1, hidden_size)
            count = attention_mask.sum(dim=1).clamp(min=1)  # avoid division by zero
            mean_pooled = sum_hidden / count  # (1, hidden_size)
            
            # Convert to numpy and normalize
            embedding = mean_pooled.cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"CodeBERT encoding error: {e}")
            return None
    
    def compute_similarity(self, code_a: str, code_b: str, max_length: int = 512) -> Optional[float]:
        """
        Compute cosine similarity between two code snippets.
        
        Returns a value in [-1, 1]. Since embeddings are L2-normalized,
        this is equivalent to the dot product.
        
        Note: This is NOT a calibrated probability of semantic equivalence.
        """
        emb_a = self.encode(code_a, max_length)
        emb_b = self.encode(code_b, max_length)
        
        if emb_a is None or emb_b is None:
            return None
        
        # Cosine similarity of normalized vectors = dot product
        similarity = float(np.dot(emb_a, emb_b))
        return similarity


def get_encoder(model_name: str = "microsoft/codebert-base", device: str = "auto") -> CodeBERTEncoder:
    """Get or create the singleton CodeBERT encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = CodeBERTEncoder(model_name=model_name, device=device)
    return _encoder_instance
