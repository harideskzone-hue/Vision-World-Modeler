import torch
import numpy as np
from typing import Optional
import logging
from shared.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class CLIPReidentifier:
    """Lightweight CLIP-based re-identification for object tracking."""
    
    def __init__(self, model_name: str = "clip-vit-base-patch32", use_mps: bool = True, lazy_load: bool = True):
        self.model_name = model_name
        self.use_mps = use_mps
        self.lazy_load = lazy_load
        self.model = None
        self.processor = None
        self.device = None
        
        if not lazy_load:
            self._initialize_model()
            
    def _initialize_model(self):
        """Load model on-demand (first embedding extraction)."""
        if self.model is not None:
            return
            
        logger.info(f"Loading CLIP model {self.model_name}...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load model from HuggingFace
            model_id = f"openai/{self.model_name.lower().replace('/', '-')}"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            
            # Device selection with M1 support and force_cpu compliance
            if DEFAULT_CONFIG.hardware.force_cpu:
                self.device = torch.device("cpu")
                logger.info("⚡ CLIP enforced to run on CPU (compliance default)")
            elif self.use_mps and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("🚀 CLIP accelerated via Apple Metal (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("🚀 CLIP accelerated via CUDA")
            else:
                self.device = torch.device("cpu")
                logger.info("CLIP running on CPU (slower)")
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Inference mode
            
            logger.info("✅ CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None

    def extract_embedding(self, image: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """
        Extract CLIP embedding for bounding box region.
        
        Args:
            image: numpy array, shape (H, W, 3), BGR format
            bbox: [x1, y1, x2, y2] coordinates
            
        Returns:
            512-dim embedding or None
        """
        if self.lazy_load:
            self._initialize_model()
            
        if self.model is None or self.processor is None:
            return None
        
        try:
            # Crop and convert BGR to RGB
            x1, y1, x2, y2 = map(int, bbox)
            
            # Boundary checks
            H, W, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return None
            
            # BGR -> RGB
            from PIL import Image
            pil_image = Image.fromarray(cropped[..., ::-1])
            
            with torch.no_grad():
                inputs = self.processor(images=pil_image, return_tensors="pt")
                
                # Move to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device)
                
                # Get image embeddings
                image_features = self.model.get_image_features(**inputs)
                
                # Handle different transformers versions
                if hasattr(image_features, "image_embeds"):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, "pooler_output"):
                    image_features = image_features.pooler_output
                elif isinstance(image_features, tuple):
                    image_features = image_features[0]
                elif not torch.is_tensor(image_features):
                    # Last resort, if it's a BaseModelOutput but hasattr fails
                    image_features = image_features[0] if hasattr(image_features, '__getitem__') else getattr(image_features, 'image_embeds', image_features)
                    
                # Normalize to unit vector
                embedding = image_features.cpu().numpy()[0]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
        
        except Exception as e:
            logger.error(f"CLIP embedding error: {e}")
            return None
    
    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract CLIP text embedding for a category or word label.
        Returns a normalized 1D numpy array.
        """
        if self.lazy_load:
            self._initialize_model()
            
        if self.model is None or self.processor is None:
            return None
            
        try:
            import torch
            with torch.no_grad():
                inputs = self.processor(text=[text], return_tensors="pt", padding=True)
                
                # Move to device
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device)
                
                # Get text embeddings
                features = self.model.get_text_features(**inputs)
                
                # Handle different transformers versions
                if hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                elif hasattr(features, 'text_embeds'):
                    features = features.text_embeds
                elif isinstance(features, tuple):
                    features = features[0]
                elif not torch.is_tensor(features):
                    features = features[0] if hasattr(features, '__getitem__') else getattr(features, 'text_embeds', features)
                    
                # Normalize to unit vector
                embedding = features.cpu().numpy()[0]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
        
        except Exception as e:
            logger.error(f"CLIP text embedding error: {e}")
            return None
            
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between normalized embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))
