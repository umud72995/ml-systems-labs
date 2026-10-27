import time
import torch
from thop import profile as thop_profile

def measure_efficiency_metrics(model, device="cpu", input_shape=(1, 1, 28, 28), warmups=20, runs=100):
    """Quantifies model parameter count, FLOPs / MACs, and microsecond-level inference latency."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape).to(device)

    # 1. Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs via THOP (MACs)
    macs, params = thop_profile(model, inputs=(dummy_input,), verbose=False)

    # 3. Latency measurement
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(dummy_input)
        
        start_time = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy_input)
        total_time = time.perf_counter() - start_time

    latency_ms = (total_time / runs) * 1000.0

    return {
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "macs": int(macs),
        "mflops": (macs * 2) / 1e6,  # 1 MAC = 2 FLOPs
        "latency_ms": latency_ms
    }
