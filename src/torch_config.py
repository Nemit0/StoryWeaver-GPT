import torch
from torch import tensor

### CUDA SETUP ###

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_device(device)
    print(f"Using {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    torch.set_default_device(device)
    print("Using CPU")

torch.set_default_dtype(torch.float64)