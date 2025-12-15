from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.config import get_settings


class CarClassifier:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.idx_to_class: Dict[int, str] = {}
        self.device: torch.device = torch.device("cpu")
        self.transform: Optional[transforms.Compose] = None
        self.tta_transforms: Optional[List[transforms.Compose]] = None
        self._loaded = False
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        settings = get_settings()
        path = Path(model_path or settings.model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.idx_to_class = checkpoint["idx_to_class"]
        num_classes = len(self.idx_to_class)
        
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        
        self._setup_transforms(settings.img_size)
        self._loaded = True
    
    def _setup_transforms(self, img_size: int) -> None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        self.tta_transforms = [
            self.transform,
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomRotation(degrees=(10, 10)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomRotation(degrees=(-10, -10)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
        ]
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def classes(self) -> List[str]:
        return list(self.idx_to_class.values())
    
    @property
    def device_name(self) -> str:
        return str(self.device)
    
    def predict(
        self,
        image_bytes: bytes,
        use_tta: bool = False
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        if use_tta:
            all_probs = []
            with torch.no_grad():
                for tf in self.tta_transforms:
                    input_tensor = tf(image).unsqueeze(0).to(self.device)
                    outputs = self.model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs)
            
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            confidence, predicted_idx = torch.max(avg_probs, 1)
            probabilities = avg_probs[0]
        else:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities.unsqueeze(0), 1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_score = confidence.item()
        
        probs_dict = {
            self.idx_to_class[i]: round(prob.item(), 4)
            for i, prob in enumerate(probabilities)
        }
        probs_dict = dict(sorted(probs_dict.items(), key=lambda x: -x[1]))
        
        return predicted_class, confidence_score, probs_dict


classifier = CarClassifier()
