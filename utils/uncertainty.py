"""
Uncertainty Quantification and Calibration

Provides uncertainty estimation for clinical safety:
- Monte Carlo Dropout for epistemic uncertainty
- Temperature scaling for calibration
- Expected Calibration Error (ECE) computation
- Out-of-Distribution (OOD) detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    model uncertainty.
    
    Args:
        model: The base model
        n_samples: Number of MC samples
        dropout_rate: Override dropout rate (if None, uses model's)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        
        # Enable dropout during inference
        if dropout_rate is not None:
            self._set_dropout_rate(dropout_rate)
    
    def _set_dropout_rate(self, rate: float):
        """Set dropout rate for all dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
    
    def _enable_dropout(self):
        """Enable dropout during eval mode."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with MC Dropout.
        
        Returns:
            mean_logits: Mean prediction across samples
            uncertainty: Standard deviation (epistemic uncertainty)
        """
        self.model.eval()
        self._enable_dropout()
        
        samples = []
        for _ in range(self.n_samples):
            if tabular is not None:
                logits = self.model(x, tabular)
            else:
                logits = self.model(x)
            samples.append(logits)
        
        samples = torch.stack(samples, dim=0)  # n_samples, B, 1
        
        mean_logits = samples.mean(dim=0)
        uncertainty = samples.std(dim=0)
        
        return mean_logits, uncertainty
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with full uncertainty breakdown.
        
        Returns:
            probs: Mean probability
            epistemic: Model uncertainty (reducible with more data)
            aleatoric: Data uncertainty (irreducible)
        """
        mean_logits, epistemic = self.forward(x, tabular)
        probs = torch.sigmoid(mean_logits)
        
        # Predictive entropy: H[E[p(y|x,w)]]
        predictive_entropy = -(probs * torch.log(probs + 1e-8) + 
                   (1-probs) * torch.log(1-probs + 1e-8))
        
        # Aleatoric = Predictive Entropy - Mutual Information
        # For MC Dropout: MI ≈ epistemic variance captures model uncertainty
        # Aleatoric = total uncertainty - epistemic uncertainty
        # We use entropy as total and std as epistemic proxy
        aleatoric = torch.clamp(predictive_entropy - epistemic, min=0.0)
        
        return probs, epistemic, aleatoric


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Learns a single temperature parameter to scale logits,
    improving probability calibration without changing predictions.
    
    Reference: "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
    ):
        """
        Learn optimal temperature from validation set.
        
        Uses NLL loss to find best temperature.
        """
        model.eval()
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, tabular, labels = batch
                    images = images.to(device)
                    tabular = tabular.to(device)
                    logits = model(images, tabular)
                else:
                    images, labels = batch
                    images = images.to(device)
                    logits = model(images)
                
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)
        
        # Optimize temperature
        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled = self.forward(all_logits)
            loss = F.binary_cross_entropy_with_logits(
                scaled.squeeze(), all_labels.float()
            )
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        print(f"Calibrated temperature: {self.temperature.item():.3f}")
        return self.temperature.item()


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error.
    
    Measures how well predicted probabilities match actual outcomes.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        probs: Predicted probabilities
        labels: True binary labels
        n_bins: Number of confidence bins
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_conf = probs[in_bin].mean()
            avg_acc = labels[in_bin].mean()
            ece += np.abs(avg_acc - avg_conf) * prop_in_bin
    
    return ece


class OODDetector(nn.Module):
    """
    Out-of-Distribution detection using energy score.
    
    Lower energy = more likely in-distribution.
    
    Reference: "Energy-based Out-of-distribution Detection"
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        threshold: float = None,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.threshold = threshold
    
    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score.
        
        E(x) = -T * log(sum(exp(f(x)/T)))
        """
        # For binary classification, treat as 2-class
        logits_2class = torch.cat([logits, -logits], dim=-1)
        return -self.temperature * torch.logsumexp(
            logits_2class / self.temperature, dim=-1
        )
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute OOD scores.
        
        Returns:
            logits: Model predictions
            energy: OOD score (higher = more likely OOD)
        """
        self.model.eval()
        
        if tabular is not None:
            logits = self.model(x, tabular)
        else:
            logits = self.model(x)
        
        energy = self.compute_energy(logits)
        
        return logits, energy
    
    def is_ood(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Check if samples are OOD.
        
        Returns:
            Boolean tensor (True = OOD)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        _, energy = self.forward(x, tabular)
        return energy > self.threshold
    
    def set_threshold(
        self,
        val_loader,
        device: torch.device,
        percentile: float = 95.0,
    ):
        """
        Set OOD threshold from validation set.
        
        Uses percentile of in-distribution energy scores.
        """
        energies = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, tabular, _ = batch
                    images = images.to(device)
                    tabular = tabular.to(device)
                    _, energy = self.forward(images, tabular)
                else:
                    images, _ = batch
                    images = images.to(device)
                    _, energy = self.forward(images)
                
                energies.append(energy.cpu())
        
        all_energies = torch.cat(energies).numpy()
        self.threshold = np.percentile(all_energies, percentile)
        
        print(f"OOD threshold set to {self.threshold:.3f} (p{percentile})")
        return self.threshold


class SafePredictor(nn.Module):
    """
    Safe prediction wrapper with uncertainty and OOD checks.
    
    Wraps a model to provide:
    - Prediction with uncertainty
    - OOD detection
    - Confidence thresholding
    - Calibrated probabilities
    """
    
    def __init__(
        self,
        model: nn.Module,
        mc_samples: int = 10,
        confidence_threshold: float = 0.85,
        uncertainty_threshold: float = 0.15,
    ):
        super().__init__()
        
        self.mc_dropout = MCDropout(model, n_samples=mc_samples)
        self.ood_detector = OODDetector(model)
        self.temp_scaling = TemperatureScaling()
        
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        tabular: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Safe prediction with full diagnostics.
        
        Returns:
            dict with: prediction, probability, uncertainty, is_ood, is_reliable
        """
        # MC Dropout prediction
        logits, uncertainty = self.mc_dropout(x, tabular)
        
        # Calibrated probability
        cal_logits = self.temp_scaling(logits)
        probs = torch.sigmoid(cal_logits)
        
        # OOD score
        _, energy = self.ood_detector(x, tabular)
        is_ood = energy > self.ood_detector.threshold if self.ood_detector.threshold else False
        
        # Reliability check
        is_reliable = (
            (uncertainty.squeeze() < self.uncertainty_threshold) &
            ((probs.squeeze() > self.confidence_threshold) | 
             (probs.squeeze() < (1 - self.confidence_threshold)))
        )
        
        return {
            'logits': logits,
            'probability': probs,
            'prediction': (probs > 0.5).float(),
            'uncertainty': uncertainty,
            'energy': energy,
            'is_ood': is_ood,
            'is_reliable': is_reliable,
        }
