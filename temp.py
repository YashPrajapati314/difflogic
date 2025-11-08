import torch

print(torch.version.cuda)
print(torch.version.__version__)
print(torch.cuda.is_available())

print(torch._C._cuda_getCompiledVersion())
print(torch._C._cuda_getDriverVersion())
