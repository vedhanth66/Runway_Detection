import torch

is_cuda_available = torch.cuda.is_available()

print(f"--- GPU Verification ---")
print(f"Is CUDA available? {is_cuda_available}")

if is_cuda_available:
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs: {device_count}")
    current_device = torch.cuda.current_device()
    print(f"Current device index: {current_device}")
    print(f"Device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available. Training will run on CPU.")

print(f"----------------------")