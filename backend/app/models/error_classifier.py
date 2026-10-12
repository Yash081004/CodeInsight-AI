"""
CodeInsight AI - Error Classifier

Fine-tunable CodeBERT sequence classifier for 5-class logic error detection.
Loads a fine-tuned checkpoint when available, otherwise reports unavailability.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import json
import logging

from backend.app.config import ERROR_LABELS, LABEL_TO_ID, ID_TO_LABEL

logger = logging.getLogger(__name__)

_classifier_instance: Optional["ErrorClassifier"] = None


class ErrorClassifier:
    """
    Five-class logic error classifier using fine-tuned CodeBERT.
    
    Classes: correct, wrong_initialization, off_by_one_loop,
             wrong_operator, wrong_condition
    """
    
    def __init__(self, model_path: Optional[Path] = None, device: str = "auto"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._trained = False
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def load(self) -> bool:
        """
        Load a fine-tuned classifier checkpoint.
        Returns True only if a trained checkpoint exists and loads successfully.
        """
        if self._loaded:
            return True
        
        if self.model_path is None or not self.model_path.exists():
            logger.info("No fine-tuned classifier checkpoint found.")
            return False
        
        # Check for required files
        config_file = self.model_path / "config.json"
        if not config_file.exists():
            logger.info(f"No config.json in {self.model_path}")
            return False
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"Loading fine-tuned classifier from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                num_labels=len(ERROR_LABELS),
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            self._trained = True
            logger.info("Fine-tuned classifier loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def is_trained(self) -> bool:
        if self._trained:
            return True
        if self.model_path and (self.model_path / "config.json").exists() and (
            (self.model_path / "model.safetensors").exists() or (self.model_path / "pytorch_model.bin").exists()
        ):
            return True
        return False
    
    def predict(self, code: str, max_length: int = 512) -> Optional[Dict]:
        """
        Predict the error class for a code snippet.
        
        Returns dict with:
            - predicted_label: str
            - probabilities: Dict[str, float]
            - confidence: float (max probability)
            
        Returns None if the model is not loaded.
        """
        if not self.load():
            return None
        
        try:
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits  # (1, num_labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            
            predicted_id = int(np.argmax(probs))
            predicted_label = ID_TO_LABEL[predicted_id]
            
            probabilities = {
                ID_TO_LABEL[i]: float(probs[i])
                for i in range(len(ERROR_LABELS))
            }
            
            return {
                "predicted_label": predicted_label,
                "probabilities": probabilities,
                "confidence": float(probs[predicted_id]),
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None


def get_classifier(model_path: Optional[Path] = None, device: str = "auto") -> ErrorClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ErrorClassifier(model_path=model_path, device=device)
    elif model_path is not None and _classifier_instance.model_path is None:
        _classifier_instance.model_path = model_path
    return _classifier_instance
