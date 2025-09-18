import torch

# Create a 2-dimensional tensor
x = torch.tensor([[2, 9, 1],
                  [7, 4, 8]])

# Find the index of the maximum value in the entire flattened tensor
index_of_max_flat = torch.argmax(x.view(-1))
print(f"Index of max in flattened tensor: {index_of_max_flat}") # Output: tensor(1) (index of 9)

# Find the indices of the maximum values along dimension 0 (columns)
indices_along_dim0 = torch.argmax(x, dim=0)
print(f"Indices along dim 0: {indices_along_dim0}") # Output: tensor([1, 0, 1]) (indices of 7, 9, 8)

# Find the indices of the maximum values along dimension 1 (rows)
indices_along_dim1 = torch.argmax(x, dim=1)
print(f"Indices along dim 1: {indices_along_dim1}") # Output: tensor([1, 2]) (indices of 9, 8)
