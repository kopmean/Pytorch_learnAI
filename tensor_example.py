import torch
import numpy as np


# 1. initialize tensor
my_tensor = torch.tensor([[1,2,3], [4,5,6]],
                         dtype=torch.float32,
                         device="mps")

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)


# 2. Tendor from numpy array
np_array = np.array([1,2,3,4,5])
tensor = torch.from_numpy(np_array)
print(tensor)

tensor = torch.tensor(np_array)
print(tensor)


# 3. Random constant
shape = (2,3)
rand_tensor = torch.rand(shape)
print(rand_tensor)

ones_tensor = torch.ones(shape)
print(ones_tensor)

zeros_tensor = torch.zeros(shape)
print(zeros_tensor)


# 4. indexing operation
print(rand_tensor)
take_row_0 = rand_tensor[0,:]
print(take_row_0)

take_00 = rand_tensor[:,0]
print(take_00)


# 5. joining tensor
concat_tensor = torch.cat([ones_tensor, zeros_tensor],
                          dim=0)
print(concat_tensor.shape)
print(ones_tensor.shape)
print(zeros_tensor.shape)

print(concat_tensor)


# 6. operation matirx
matrix_A = torch.rand((3,3))
column_vector = torch.rand((3,1))
print(matrix_A)
print(column_vector)
out = torch.matmul(matrix_A, column_vector)
print(out)


# 7. element-wise operation
ones_tensor = torch.ones((2,3))
twos_tensor = 2 * ones_tensor
print(ones_tensor)
print(twos_tensor)


# 8. broadcasting
# https://numpy.org/doc/stable/user/basics.broadcasting.html
x1 = torch.rand((5,5))
x2 = torch.zeros((1,5))

out = x1 * x2
print(x1)
print(x2)
print(out)


# 9. tensor reshape
x = torch.arange(9)
print(x)
x_3x3 = x.reshape(3,3)
print(x_3x3)


# gpu or cpu or apple silicon
import platform
import re

this_device = platform.platform()
if torch.cuda.is_available():
    device = "cuda"
elif re.search("arm64", this_device):
    device = "mps"
else:
    device = "gpu"

print(device)
print(this_device)
print(platform.system())
print(platform.machine())

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")
print(mps_device)
my_tensor = torch.tensor([[1,2,3]], dtype=torch.float32,
                        device=mps_device)
print(my_tensor)
