"""
MambaDerm Unified Clinical Prediction System

Ensemble prediction combining:
- ISIC 2024 model (Binary: Malignant vs Benign)
- HAM10000 model (7-class: specific diagnosis)

Provides BOTH:
1. Malignancy assessment (from both models)
2. Specific diagnosis (7-class)
3. Combined confidence score

Usage:
    python predict_unified.py --image lesion.jpg
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import MambaDerm
from data.transforms import get_val_transforms, get_tta_transforms
from data.ham10000_dataset import HAM10000_CLASSES, HAM10000_CLASS_NAMES


# HAM10000 class → Malignancy mapping
HAM10000_MALIGNANCY = {
    'akiec': 'PRE_MALIGNANT',  # Actinic keratoses - pre-cancerous
    'bcc': 'MALIGNANT',        # Basal cell carcinoma
    'bkl': 'BENIGN',           # Benign keratosis
    'df': 'BENIGN',            # Dermatofibroma
    'mel': 'MALIGNANT',        # Melanoma - most dangerous!
    'nv': 'BENIGN',            # Melanocytic nevi
    'vasc': 'BENIGN',          # Vascular lesions
}

# Malignancy weight for ensemble (how much each HAM class contributes to malignancy score)
HAM10000_MALIGNANCY_WEIGHT = {
    'akiec': 0.6,  # Pre-malignant
    'bcc': 0.9,    # Malignant (but less aggressive than melanoma)
    'bkl': 0.1,    # Benign
    'df': 0.05,    # Benign
    'mel': 1.0,    # Most malignant
    'nv': 0.05,    # Benign
    'vasc': 0.05,  # Benign
}

# Risk levels
RISK_LEVELS = {
    'CRITICAL': {'threshold': 0.8, 'emoji': '🔴🔴', 'action': 'IMMEDIATE'},
    'HIGH': {'threshold': 0.6, 'emoji': '🔴', 'action': 'URGENT'},
    'MODERATE': {'threshold': 0.3, 'emoji': '🟡', 'action': 'SOON'},
    'LOW': {'threshold': 0.0, 'emoji': '🟢', 'action': 'ROUTINE'},
}

# Clinical recommendations
CLINICAL_ACTIONS = {
    'IMMEDIATE': "⚠️ CRITICAL: Suspected melanoma or aggressive carcinoma. IMMEDIATE biopsy and oncology referral required.",
    'URGENT': "⚠️ URGENT: High malignancy probability. Dermatologist evaluation within 24-48 hours.",
    'SOON': "⚠️ ATTENTION: Elevated risk or pre-malignant. Dermatologist evaluation within 1-2 weeks.",
    'ROUTINE': "✓ Low concern: Likely benign. Monitor changes. Annual skin examination recommended.",
}


class UnifiedClinicalPredictor:
    """
    Ensemble prediction using ISIC 2024 + HAM10000 models.
    
    Combines:
    - Binary malignancy assessment (ISIC 2024)
    - Specific 7-class diagnosis (HAM10000)
    - Weighted ensemble for robust malignancy score
    
    Args:
        isic_checkpoint: Path to ISIC 2024 model checkpoint
        ham_checkpoint: Path to HAM10000 model checkpoint
        device: 'cuda', 'cpu', or 'auto'
        use_tta: Enable test-time augmentation
        ensemble_weight: Weight for ISIC model (0-1), HAM gets 1-weight
    """
    
    def __init__(
        self,
        isic_checkpoint: Union[str, Path] = 'checkpoints/fold_0/best_model.pt',
        ham_checkpoint: Union[str, Path] = 'checkpoints_ham10000/fold_0/best_model.pt',
        device: str = 'auto',
        use_tta: bool = True,
        tta_transforms: int = 5,
        ensemble_weight: float = 0.5,  # 50% ISIC, 50% HAM-derived
        img_size: int = 224,
    ):
        self.device = self._get_device(device)
        self.use_tta = use_tta
        self.tta_transforms = tta_transforms
        self.ensemble_weight = ensemble_weight
        self.img_size = img_size
        
        # Load both models
        print("=" * 60)
        print("🔬 MAMBADERM UNIFIED CLINICAL PREDICTOR")
        print("=" * 60)
        
        print("\n📦 Loading ISIC 2024 model (Binary)...")
        self.isic_model = self._load_isic_model(isic_checkpoint)
        
        print("\n📦 Loading HAM10000 model (7-class)...")
        self.ham_model = self._load_ham_model(ham_checkpoint)
        
        # Transforms
        self.val_transform = get_val_transforms(img_size)
        self.tta_transform_list = get_tta_transforms(img_size) if use_tta else None
        
        # Class info
        self.ham_class_names = list(HAM10000_CLASSES.keys())
        self.ham_display_names = HAM10000_CLASS_NAMES
        
        print(f"\n✓ Unified Predictor Ready")
        print(f"  Device: {self.device}")
        print(f"  Ensemble: {ensemble_weight*100:.0f}% ISIC + {(1-ensemble_weight)*100:.0f}% HAM")
        print(f"  TTA: {'Enabled' if use_tta else 'Disabled'}")
        print("=" * 60)
    
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_isic_model(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        """Load ISIC 2024 binary classification model."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"  ⚠️ ISIC checkpoint not found: {checkpoint_path}")
            return None
        
        model = MambaDerm(
            img_size=self.img_size,
            embed_dim=96,
            depths=[2, 2, 4],
            d_state=16,
            num_numerical_features=34,
            num_categorical_features=6,
            num_classes=1,  # Binary
            dropout=0.1,
            drop_path_rate=0.1,
            use_tabular=True,
            use_multi_scale=True,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  ✓ Loaded from {checkpoint_path}")
        return model
    
    def _load_ham_model(self, checkpoint_path: Union[str, Path]) -> nn.Module:
        """Load HAM10000 7-class model."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"  ⚠️ HAM10000 checkpoint not found: {checkpoint_path}")
            return None
        
        model = MambaDerm(
            img_size=self.img_size,
            embed_dim=96,
            depths=[2, 2, 4],
            d_state=16,
            num_numerical_features=19,
            num_categorical_features=0,
            num_classes=7,  # 7-class
            dropout=0.1,
            drop_path_rate=0.1,
            use_tabular=True,
            use_multi_scale=True,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  ✓ Loaded from {checkpoint_path}")
        return model
    
    def _preprocess_image(self, image_np: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference."""
        transformed = self.val_transform(image=image_np)
        return transformed['image']
    
    def _get_isic_tabular(self) -> torch.Tensor:
        """Default tabular for ISIC model."""
        return torch.zeros(1, 40)
    
    def _get_ham_tabular(self) -> torch.Tensor:
        """Default tabular for HAM model."""
        return torch.zeros(1, 19)
    
    @torch.no_grad()
    def _predict_isic(self, image_np: np.ndarray) -> float:
        """Get malignancy probability from ISIC model."""
        if self.isic_model is None:
            return None
        
        if self.use_tta:
            all_probs = []
            for transform in self.tta_transform_list[:self.tta_transforms]:
                transformed = transform(image=image_np)
                img = transformed['image'].unsqueeze(0).to(self.device)
                tabular = self._get_isic_tabular().to(self.device)
                logits = self.isic_model(img, tabular)
                prob = torch.sigmoid(logits).item()
                all_probs.append(prob)
            return np.mean(all_probs)
        else:
            img = self._preprocess_image(image_np).unsqueeze(0).to(self.device)
            tabular = self._get_isic_tabular().to(self.device)
            logits = self.isic_model(img, tabular)
            return torch.sigmoid(logits).item()
    
    @torch.no_grad()
    def _predict_ham(self, image_np: np.ndarray) -> np.ndarray:
        """Get 7-class probabilities from HAM model."""
        if self.ham_model is None:
            return None
        
        if self.use_tta:
            all_probs = []
            for transform in self.tta_transform_list[:self.tta_transforms]:
                transformed = transform(image=image_np)
                img = transformed['image'].unsqueeze(0).to(self.device)
                tabular = self._get_ham_tabular().to(self.device)
                logits = self.ham_model(img, tabular)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                all_probs.append(probs)
            return np.mean(all_probs, axis=0)
        else:
            img = self._preprocess_image(image_np).unsqueeze(0).to(self.device)
            tabular = self._get_ham_tabular().to(self.device)
            logits = self.ham_model(img, tabular)
            return F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    def _compute_ham_malignancy(self, ham_probs: np.ndarray) -> float:
        """Compute malignancy score from HAM10000 7-class probabilities."""
        malignancy_score = 0.0
        for i, class_name in enumerate(self.ham_class_names):
            weight = HAM10000_MALIGNANCY_WEIGHT[class_name]
            malignancy_score += ham_probs[i] * weight
        return malignancy_score
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict:
        """
        Unified prediction combining both models.
        
        Returns:
            Dictionary with:
            - ensemble_malignancy: Combined malignancy score
            - isic_malignancy: ISIC 2024 model malignancy
            - ham_malignancy: HAM-derived malignancy
            - specific_diagnosis: Most likely HAM10000 class
            - all_diagnoses: All 7-class probabilities
            - risk_level: Clinical risk assessment
            - recommendation: Clinical action
        """
        start_time = time.time()
        
        # Load image
        if isinstance(image, (str, Path)):
            image_np = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Get predictions from both models
        isic_prob = self._predict_isic(image_np)
        ham_probs = self._predict_ham(image_np)
        
        # Compute HAM-derived malignancy
        ham_malignancy = self._compute_ham_malignancy(ham_probs) if ham_probs is not None else None
        
        # Ensemble malignancy score
        if isic_prob is not None and ham_malignancy is not None:
            ensemble_malignancy = (
                self.ensemble_weight * isic_prob + 
                (1 - self.ensemble_weight) * ham_malignancy
            )
        elif isic_prob is not None:
            ensemble_malignancy = isic_prob
        elif ham_malignancy is not None:
            ensemble_malignancy = ham_malignancy
        else:
            raise RuntimeError("No models available for prediction")
        
        # Specific diagnosis from HAM
        if ham_probs is not None:
            pred_idx = np.argmax(ham_probs)
            specific_class = self.ham_class_names[pred_idx]
            specific_display = self.ham_display_names[pred_idx]
            specific_confidence = ham_probs[pred_idx]
            specific_type = HAM10000_MALIGNANCY[specific_class]
        else:
            specific_class = None
            specific_display = "Unknown"
            specific_confidence = 0.0
            specific_type = "UNKNOWN"
        
        # Risk assessment
        risk_level = 'LOW'
        for level, info in RISK_LEVELS.items():
            if ensemble_malignancy >= info['threshold']:
                risk_level = level
                break
        
        risk_info = RISK_LEVELS[risk_level]
        recommendation = CLINICAL_ACTIONS[risk_info['action']]
        
        inference_time = time.time() - start_time
        
        result = {
            'ensemble': {
                'malignancy_score': float(ensemble_malignancy),
                'malignancy_pct': f"{ensemble_malignancy * 100:.1f}%",
                'is_malignant': ensemble_malignancy >= 0.5,
            },
            'isic_model': {
                'malignancy': float(isic_prob) if isic_prob else None,
                'available': isic_prob is not None,
            },
            'ham_model': {
                'malignancy_derived': float(ham_malignancy) if ham_malignancy else None,
                'available': ham_probs is not None,
            },
            'specific_diagnosis': {
                'class_code': specific_class,
                'class_name': specific_display,
                'confidence': float(specific_confidence),
                'confidence_pct': f"{specific_confidence * 100:.1f}%",
                'type': specific_type,
            },
            'all_diagnoses': {
                self.ham_class_names[i]: {
                    'probability': float(ham_probs[i]),
                    'display_name': self.ham_display_names[i],
                    'type': HAM10000_MALIGNANCY[self.ham_class_names[i]],
                }
                for i in range(7)
            } if ham_probs is not None else None,
            'clinical': {
                'risk_level': risk_level,
                'risk_emoji': risk_info['emoji'],
                'action_required': risk_info['action'],
                'recommendation': recommendation,
            },
            'metadata': {
                'inference_time_ms': f"{inference_time * 1000:.1f}",
                'device': str(self.device),
                'tta_enabled': self.use_tta,
                'ensemble_weight': self.ensemble_weight,
            },
        }
        
        return result


def print_unified_prediction(result: Dict):
    """Pretty print unified prediction."""
    ensemble = result['ensemble']
    specific = result['specific_diagnosis']
    clinical = result['clinical']
    meta = result['metadata']
    
    print("\n" + "=" * 70)
    print("🔬 MAMBADERM UNIFIED CLINICAL PREDICTION")
    print("   Powered by ISIC 2024 + HAM10000 Ensemble")
    print("=" * 70)
    
    # Malignancy Assessment
    mal_score = ensemble['malignancy_score']
    bar_filled = int(mal_score * 40)
    bar = '█' * bar_filled + '░' * (40 - bar_filled)
    
    print(f"\n🎯 MALIGNANCY ASSESSMENT")
    print(f"   Benign [{bar}] Malignant")
    print(f"   Ensemble Score: {ensemble['malignancy_pct']} {'⚠️ MALIGNANT' if ensemble['is_malignant'] else '✓ BENIGN'}")
    
    # Model contributions
    isic = result['isic_model']
    ham = result['ham_model']
    if isic['available']:
        print(f"   └─ ISIC 2024 Model:  {isic['malignancy']*100:.1f}%")
    if ham['available']:
        print(f"   └─ HAM10000 Derived: {ham['malignancy_derived']*100:.1f}%")
    
    # Specific Diagnosis
    print(f"\n📋 SPECIFIC DIAGNOSIS")
    type_emoji = {'MALIGNANT': '🔴', 'PRE_MALIGNANT': '🟡', 'BENIGN': '🟢'}
    print(f"   {type_emoji.get(specific['type'], '⚪')} {specific['class_name']}")
    print(f"   Confidence: {specific['confidence_pct']}")
    print(f"   Category: {specific['type']}")
    
    # All probabilities
    if result['all_diagnoses']:
        print(f"\n📊 ALL CLASS PROBABILITIES")
        sorted_diag = sorted(
            result['all_diagnoses'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        for code, info in sorted_diag:
            prob = info['probability']
            bar_len = int(prob * 25)
            type_icon = {'MALIGNANT': '🔴', 'PRE_MALIGNANT': '🟡', 'BENIGN': '🟢'}[info['type']]
            bar = '█' * bar_len + '░' * (25 - bar_len)
            print(f"   {type_icon} {info['display_name'][:22]:22s} [{bar}] {prob*100:5.1f}%")
    
    # Clinical Assessment
    print(f"\n🏥 CLINICAL ASSESSMENT")
    print(f"   Risk Level: {clinical['risk_emoji']} {clinical['risk_level']}")
    print(f"   Action: {clinical['action_required']}")
    print(f"\n💊 RECOMMENDATION:")
    print(f"   {clinical['recommendation']}")
    
    # Metadata
    print(f"\n⏱️  Inference: {meta['inference_time_ms']}ms | Device: {meta['device']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MambaDerm Unified Clinical Prediction (ISIC + HAM10000 Ensemble)",
    )
    
    # Input
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    
    # Model checkpoints
    parser.add_argument('--isic_checkpoint', type=str,
                        default='checkpoints/fold_0/best_model.pt')
    parser.add_argument('--ham_checkpoint', type=str,
                        default='checkpoints_ham10000/fold_0/best_model.pt')
    
    # Options
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--ensemble_weight', type=float, default=0.5,
                        help='Weight for ISIC model (0-1)')
    parser.add_argument('--json', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = UnifiedClinicalPredictor(
        isic_checkpoint=args.isic_checkpoint,
        ham_checkpoint=args.ham_checkpoint,
        device=args.device,
        use_tta=not args.no_tta,
        ensemble_weight=args.ensemble_weight,
    )
    
    # Predict
    result = predictor.predict(args.image)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_unified_prediction(result)


if __name__ == "__main__":
    main()
