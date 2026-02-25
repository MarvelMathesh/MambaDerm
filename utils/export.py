"""
Model Export Utilities

Provides export functionality for deployment:
- TorchScript compilation
- ONNX export
- INT8 quantization
- Inference benchmarking
"""

import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np


def export_torchscript(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
    example_tabular_dim: int = 39,
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: The model to export
        save_path: Path to save the model
        img_size: Input image size
        example_tabular_dim: Tabular feature dimension
        
    Returns:
        Path to saved model
    """
    model.eval()
    
    # Create example inputs
    example_image = torch.randn(1, 3, img_size, img_size)
    example_tabular = torch.randn(1, example_tabular_dim)
    
    # Script the model
    try:
        scripted = torch.jit.script(model)
    except Exception:
        # Fall back to tracing if scripting fails
        print("Scripting failed, using tracing...")
        scripted = torch.jit.trace(
            model, 
            (example_image, example_tabular),
            strict=False
        )
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(save_path))
    
    print(f"TorchScript model saved to: {save_path}")
    return str(save_path)


def export_onnx(
    model: nn.Module,
    save_path: str,
    img_size: int = 224,
    example_tabular_dim: int = 39,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: The model to export
        save_path: Path to save the model
        img_size: Input image size
        example_tabular_dim: Tabular feature dimension
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        
    Returns:
        Path to saved model
    """
    model.eval()
    
    # Create example inputs
    example_image = torch.randn(1, 3, img_size, img_size)
    example_tabular = torch.randn(1, example_tabular_dim)
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'image': {0: 'batch'},
            'tabular': {0: 'batch'},
            'output': {0: 'batch'},
        }
    
    # Export
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        (example_image, example_tabular),
        str(save_path),
        input_names=['image', 'tabular'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"ONNX model saved to: {save_path}")
    return str(save_path)


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Quantizes linear layers to INT8 for faster CPU inference.
    
    Args:
        model: The model to quantize
        dtype: Quantization dtype (qint8 or float16)
        
    Returns:
        Quantized model
    """
    model.eval()
    
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
    )
    
    # Count size reduction
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quant_size = sum(p.numel() * p.element_size() for p in quantized.parameters())
    
    print(f"Original size: {orig_size / 1e6:.2f} MB")
    print(f"Quantized size: {quant_size / 1e6:.2f} MB")
    print(f"Compression ratio: {orig_size / quant_size:.2f}x")
    
    return quantized


def benchmark_inference(
    model: nn.Module,
    img_size: int = 224,
    tabular_dim: int = 39,
    batch_sizes: list = [1, 4, 8, 16],
    n_warmup: int = 10,
    n_runs: int = 100,
    device: str = 'cuda',
) -> dict:
    """
    Benchmark model inference latency and throughput.
    
    Args:
        model: The model to benchmark
        img_size: Input image size
        tabular_dim: Tabular feature dimension
        batch_sizes: Batch sizes to test
        n_warmup: Warmup iterations
        n_runs: Benchmark iterations
        device: Device to benchmark on
        
    Returns:
        Dict with latency and throughput results
    """
    model.eval()
    model = model.to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create inputs
        images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        tabular = torch.randn(batch_size, tabular_dim).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(images, tabular)
        
        # Synchronize
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(images, tabular)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        results[batch_size] = {
            'mean_latency_ms': latencies.mean(),
            'std_latency_ms': latencies.std(),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'throughput_img_per_sec': batch_size * 1000 / latencies.mean(),
        }
        
        print(f"Batch {batch_size}: {latencies.mean():.2f}ms (±{latencies.std():.2f}), "
              f"{results[batch_size]['throughput_img_per_sec']:.1f} img/s")
    
    return results


class InferencePipeline:
    """
    Complete inference pipeline for deployment.
    
    Handles preprocessing, inference, and postprocessing.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_amp: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        # Load model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Normalization (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images."""
        return (images - self.mean) / self.std
    
    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run inference.
        
        Args:
            images: (B, 3, H, W) normalized images
            tabular: (B, F) tabular features
            
        Returns:
            Probabilities (B,)
        """
        images = images.to(self.device)
        tabular = tabular.to(self.device)
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                logits = self.model(images, tabular)
        else:
            logits = self.model(images, tabular)
        
        probs = torch.sigmoid(logits.squeeze())
        return probs.cpu()


if __name__ == "__main__":
    print("Export utilities loaded successfully")
