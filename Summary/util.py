from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

print(device)

print(cuda.is_available())

print(cuda.get_device_name())

