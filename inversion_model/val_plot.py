import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 强制使用系统浏览器进行渲染
import plotly.io as pio
pio.renderers.default = "browser"

# Load two datasets
data1 = np.load('./data/referenceData/frc_matrix6580.npy')
# data1 = np.load('./predictdata/863predicted_test_output7_3.npy')
data2_full = np.load('./data/predictdata/6580predicted_test_output_EC.npy')
# data2 = np.where(data2 > 0.4, 1, 0)

# 阈值化参考数据用于第一幅图（蓝色点）
data1 = np.where(data1 > 0.5, 1, 0)
# 第二幅图用原始数值着色，这里仅用掩码筛选显示的点
t2_thresh = 0.4
mask2 = data2_full > t2_thresh

# Get the coordinates of the non-zero (or significant) points for both datasets
x1, y1, z1 = np.nonzero(data1)
x2, y2, z2 = np.nonzero(mask2)
# 取对应点的原始数值作为颜色
colors2 = data2_full[mask2].astype(float)
cmin2 = float(colors2.min()) if colors2.size > 0 else 0.0
cmax2 = float(colors2.max()) if colors2.size > 0 else 1.0

# Create traces for each dataset
trace1 = go.Scatter3d(
    x=x1, y=y1, z=z1,
    mode='markers',
    marker=dict(
        size=1,
        color='blue',  # Color for the first dataset
        opacity=0.8
    ),
    name='Dataset 1'
)

trace2 = go.Scatter3d(
    x=x2, y=y2, z=z2,
    mode='markers',
    marker=dict(
        size=1,
        color=colors2,                 # 使用原始数值着色
        colorscale='Viridis',          # 颜色映射
        cmin=cmin2,
        cmax=cmax2,
        showscale=True,                # 显示颜色条
        colorbar=dict(title='Value'),
        opacity=0.8
    ),
    name='Dataset 2'
)

# 创建左右两个 3D 子图
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=['Dataset 1', 'Dataset 2'],
    horizontal_spacing=0.05
)

# 将各自的 trace 放入对应子图
fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)

# 更新两个子图的坐标轴范围与纵横比，保持一致
fig.update_layout(
    title="3D Point Cloud of Two Datasets (Side-by-Side)",
    width=1200,
    height=500,
    scene=dict(
        xaxis=dict(title='X Axis', range=[0, 96], backgroundcolor="rgb(200, 200, 230)"),
        yaxis=dict(title='Y Axis', range=[0, 72], backgroundcolor="rgb(230, 200, 230)"),
        zaxis=dict(title='Z Axis', range=[0, 24], backgroundcolor="rgb(230, 230, 200)"),
        aspectmode='manual',
        aspectratio=dict(x=4, y=3, z=1),
    ),
    scene2=dict(
        xaxis=dict(title='X Axis', range=[0, 96], backgroundcolor="rgb(200, 200, 230)"),
        yaxis=dict(title='Y Axis', range=[0, 72], backgroundcolor="rgb(230, 200, 230)"),
        zaxis=dict(title='Z Axis', range=[0, 24], backgroundcolor="rgb(230, 230, 200)"),
        aspectmode='manual',
        aspectratio=dict(x=4, y=3, z=1),
    ),
)

# Show the plot
fig.show()