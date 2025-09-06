3D UNet - Minimal Dependencies (with versions)

Tested environment:
- Python 3.9

Required packages:
- torch == 2.1.2
- numpy == 1.24.4
- matplotlib == 3.7.3
- plotly == 5.18.0

Quick install (CPU example):
- pip install --index-url https://download.pytorch.org/whl/cpu torch==2.1.2
- pip install numpy==1.24.4 matplotlib==3.7.3 plotly==5.18.0

Notes:
- If you need GPU acceleration, install the CUDA-matched PyTorch build from the official source and keep the same versions for the remaining packages.
- These versions are known to work with Python 3.9 and the current codebase; newer versions may also work.