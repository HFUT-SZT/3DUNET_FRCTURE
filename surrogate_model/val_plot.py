import numpy as np
import matplotlib.pyplot as plt

def denormalize_matrix(M_norm, min_value, max_value):
    """
    将归一化后的三维矩阵还原为原始矩阵
    参数:
    - M_norm: 归一化后的三维矩阵 (numpy 数组)
    - min_value: 原始数据的最小值
    - max_value: 原始数据的最大值
    
    返回:
    - M_original: 还原后的原始矩阵
    """
    M_original = M_norm * (max_value - min_value) + min_value
    return M_original

def pressure_to_head(P, rho=1000, g=9.81):
    """
    将压力转换为水头
    参数:
    - P: 压力矩阵（单位：帕斯卡，Pa）
    - rho: 水的密度（kg/m³，默认1000 kg/m³）
    - g: 重力加速度（m/s²，默认9.81 m/s²）
    
    返回:
    - H: 水头矩阵（单位：米，m）
    """
    H = P / (rho * g)
    return H

def calculate_rmse(predicted, actual):
    """
    计算均方根误差 (RMSE)
    参数:
    - predicted: 预测矩阵
    - actual: 实际矩阵
    
    返回:
    - RMSE 值
    """
    diff = predicted - actual
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

# 加载归一化后的矩阵
M_norm1 = np.load('./data/predictdata/output6382_2.npy')  # 预测数据
M_norm2 = np.load('./data/referenceData/normalized_pressure6382_2.npy')  # 预测数据

# 假设归一化时使用的最小值和最大值
min_value = -57565.62595383519
max_value = 33239.17167432396

# 还原矩阵（假设这些是压力数据）
M_original1 = denormalize_matrix(M_norm1, min_value, max_value)  # 预测数据的压力
M_original2 = denormalize_matrix(M_norm2, min_value, max_value)  # 实际数据的压力

# 将压力转换为水头
H_original1 = pressure_to_head(M_original1)
H_original2 = pressure_to_head(M_original2)

# 获取Z方向的切片（假设Z轴是第三维度，即矩阵的最后一维）
slice_index = M_original1.shape[2] // 2  # 获取Z方向的中间切片

# 计算 RMSE
rmse_value = calculate_rmse(H_original1[:, :, slice_index], H_original2[:, :, slice_index])

# 绘制两个水头切片及其差值
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制 H_original1（预测水头）的切片
ax = axes[0]
img1 = ax.imshow(H_original1[:, :, slice_index], cmap='jet', origin='lower')
ax.set_title('Predicted Head Slice (Z={})'.format(slice_index))
ax.axis('off')
fig.colorbar(img1, ax=ax, orientation='vertical')  # 添加颜色条
img1.set_clim(-1, 3)
# 绘制 H_original2（实际水头）的切片
ax = axes[1]
img2 = ax.imshow(H_original2[:, :, slice_index], cmap='jet', origin='lower')
ax.set_title('Real Head Slice (Z={})'.format(slice_index))
ax.axis('off')
fig.colorbar(img2, ax=ax, orientation='vertical')  # 添加颜色条
img2.set_clim(-1, 3)
# 绘制两者差值的切片（预测水头 - 实际水头）
ax = axes[2]
difference = H_original1[:, :, slice_index] - H_original2[:, :, slice_index]
img3 = ax.imshow(difference, cmap='jet', origin='lower')
# 设置颜色条的范围为[-0.5, 0.5]
img3.set_clim(-1, 1)
fig.colorbar(img3, ax=ax, orientation='vertical')  # 添加颜色条
ax.set_title('Difference (Predicted - Real) Head\nRMSE: {:.3f}'.format(rmse_value))
ax.axis('off')

plt.tight_layout()
plt.show()