"""
GPU Detection Script - Check if PyTorch can access your RTX 3050
"""
import torch

print("="*60)
print("GPU DETECTION")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\n✅ GPU is ready to use!")
else:
    print(f"\n❌ CUDA not available. PyTorch will use CPU.")
    print(f"\nTo enable GPU:")
    print(f"1. Uninstall current PyTorch:")
    print(f'   pip uninstall torch torchvision torchaudio')
    print(f"2. Install PyTorch with CUDA 11.8:")
    print(f'   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')

print("="*60)
