import torch
print(f"PyTorch version: {torch.__version__}")
print(torch.version.cuda)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current GPU device: {torch.cuda.get_device_name(0)}")