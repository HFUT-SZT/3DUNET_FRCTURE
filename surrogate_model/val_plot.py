import numpy as np
import matplotlib.pyplot as plt

def denormalize_matrix(M_norm, min_value, max_value):
    """
    Restore the original 3D matrix from its normalized form
    Args:
    - M_norm: normalized 3D matrix (numpy array)
    - min_value: minimum value of the original data
    - max_value: maximum value of the original data
    
    Returns:
    - M_original: denormalized original matrix
    """
    M_original = M_norm * (max_value - min_value) + min_value
    return M_original

def pressure_to_head(P, rho=1000, g=9.81):
    """
    Convert pressure to hydraulic head
    Args:
    - P: pressure matrix (Pa)
    - rho: water density (kg/m³, default 1000 kg/m³)
    - g: gravitational acceleration (m/s², default 9.81 m/s²)
    
    Returns:
    - H: hydraulic head matrix (m)
    """
    H = P / (rho * g)
    return H

def calculate_rmse(predicted, actual):
    """
    Compute Root Mean Square Error (RMSE)
    Args:
    - predicted: predicted matrix
    - actual: ground-truth matrix
    
    Returns:
    - RMSE value
    """
    diff = predicted - actual
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

# Load normalized matrices  （group2）
M_norm1 = np.load('./data/predictdata/output6400_2.npy')  # predicted data
M_norm2 = np.load('./data/referenceData/normalized_pressure6400_2.npy')  # reference data

# global_min1 = -57574.703230342304
# global_max1 = 33487.964979615776
# global_min2 = -57565.62595383519
# global_max2 = 33239.17167432396
# global_min3 = -54876.42604464349
# global_max3 = 161509.1730952983

# Min and max used during normalization (assumed) group2
min_value = -57565.62595383519
max_value = 33239.17167432396



# Denormalize to original pressure matrices
M_original1 = denormalize_matrix(M_norm1, min_value, max_value)  # predicted pressure
M_original2 = denormalize_matrix(M_norm2, min_value, max_value)  # ground-truth pressure

# Convert pressure to hydraulic head
H_original1 = pressure_to_head(M_original1)
H_original2 = pressure_to_head(M_original2)

# Get slice along the Z direction (assume Z is the last axis)
slice_index = M_original1.shape[2] // 2  # middle slice along Z

# Compute RMSE
rmse_value = calculate_rmse(H_original1[:, :, slice_index], H_original2[:, :, slice_index])

# Plot two head slices and their difference
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot slice of H_original1 (predicted head)
ax = axes[0]
img1 = ax.imshow(H_original1[:, :, slice_index], cmap='jet', origin='lower')
ax.set_title('Predicted Head Slice (Z={})'.format(slice_index))
ax.axis('off')
fig.colorbar(img1, ax=ax, orientation='vertical')  # add colorbar
img1.set_clim(-1, 3)
# Plot slice of H_original2 (real head)
ax = axes[1]
img2 = ax.imshow(H_original2[:, :, slice_index], cmap='jet', origin='lower')
ax.set_title('Real Head Slice (Z={})'.format(slice_index))
ax.axis('off')
fig.colorbar(img2, ax=ax, orientation='vertical')  # add colorbar
img2.set_clim(-1, 3)
# Plot slice of the difference (predicted head - real head)
ax = axes[2]
difference = H_original1[:, :, slice_index] - H_original2[:, :, slice_index]
img3 = ax.imshow(difference, cmap='jet', origin='lower')
# Set colorbar range to [-1, 1]
img3.set_clim(-1, 1)
fig.colorbar(img3, ax=ax, orientation='vertical')  # add colorbar
ax.set_title('Difference (Predicted - Real) Head\nRMSE: {:.3f}'.format(rmse_value))
ax.axis('off')

plt.tight_layout()
plt.show()