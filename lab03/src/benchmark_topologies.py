import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topologies.mlp_topology import UmudDenseNetwork
from topologies.cnn_topology import UmudConvClassifier
from topologies.vit_topology import UmudPatchTransformer
from topologies.cost_evaluator import measure_efficiency_metrics

def execute_benchmarks():
    print("[INFO] Initializing Umud Kalbizada Model Topologies...")
    models = {
        "Dense MLP (3-Stage)": UmudDenseNetwork(),
        "Convolutional Net (2-Stage)": UmudConvClassifier(),
        "Vision Transformer (Patch-16)": UmudPatchTransformer()
    }

    metrics = {}
    for name, m in models.items():
        res = measure_efficiency_metrics(m)
        metrics[name] = res
        print(f"[{name}] Params: {res['parameters']:,} | MACs: {res['macs']:,} | Latency: {res['latency_ms']:.3f} ms")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "systems_benchmarks.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SUCCESS] Metrics exported to {results_dir}")

if __name__ == "__main__":
    execute_benchmarks()
