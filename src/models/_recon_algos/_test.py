import numpy as np
# import magni.imaging.visualisation
from interp_reconstructions import linear_interpolation
from scipy.interpolate import griddata

# TODO: perform linear interpolation for each channel; stack up results
original_array = np.random.rand(3, 224, 224)
# Extract one channel for demonstration
channel_data = original_array[0, :, :]  # shape: (224, 224)

# Create a boolean mask that is True where data is kept and False where data is masked
mask = np.zeros_like(channel_data, dtype=bool)
mask[::2, :] = True  # keep every other row

# Verify how many values were "kept" vs. masked
# num_pixels_total = channel_data.size
# num_pixels_kept = mask.sum()
# num_pixels_masked = num_pixels_total - num_pixels_kept
# print(f"Total pixels: {num_pixels_total}, Kept: {num_pixels_kept}, Masked: {num_pixels_masked}")

# Coordinates of unmasked (kept) pixels
rows, cols = np.where(mask)
unique_coords = np.column_stack((rows, cols))  # shape: (#kept_pixels, 2)

# The pixel values at these coordinates
measurements = channel_data[mask]  # shape: (#kept_pixels,)

# For simplicity, use the identity transform.
Psi = np.eye(channel_data.size)

# Build the dictionary
var = {
    "h": channel_data.shape[0],  # 224
    "w": channel_data.shape[1],  # 224
    "unique_coords": unique_coords,
    "measurements": measurements,
    "Psi": Psi,
}

# Step 5: Interpolate
reconstructed_vec, reconstruction_time, coeffs, _ = linear_interpolation(var)

print(f"Reconstruction time: {reconstruction_time:.4f} seconds")
print("Reconstructed image vector shape:", reconstructed_vec.shape)
print("Reconstructed coefficients vector shape:", coeffs.shape)

reconstructed_2d = reconstructed_vec.reshape(channel_data.shape)

print(reconstructed_2d.shape)