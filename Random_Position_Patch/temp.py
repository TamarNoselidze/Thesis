import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate the noise vectors
fixed_noises = [torch.randn(1, 100, 1, 1).to(device) for _ in range(5)]

# Save to a file (save as a list of tensors)
torch.save(fixed_noises, "fixed_noises.pt")