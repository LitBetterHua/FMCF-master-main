import torch

cuda_available = torch.cuda.is_available()

if cuda_available:
    cuda_version = torch.version.cuda
    cuda_path = torch.__file__.split('lib')[0]
    print("CUDA version:", cuda_version)
    print("CUDA installation path:", cuda_path)
else:
    print("CUDA is not available on this system.")
