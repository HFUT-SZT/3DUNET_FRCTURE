import torch
import numpy as np
import time
from model1_out3 import UNet3D


def load_model(model_path, out_channels=3):
    model = UNet3D(out_channels=out_channels)
    # 始终在CPU上加载权重，避免设备不一致
    state = torch.load(model_path, map_location=torch.device('cpu'))

    # 过滤掉最后一层(Conv3d)的权重与偏置，先加载其余权重
    filtered_state = {k: v for k, v in state.items() if not k.startswith('final_conv.0.')}
    _load_result = model.load_state_dict(filtered_state, strict=False)
    # 可根据需要打印检查: print(_load_result)

    # 将原pth最后一层的前out_channels(=3)通道权重复制到新模型最后一层
    with torch.no_grad():
        if 'final_conv.0.weight' in state:
            pretrained_w = state['final_conv.0.weight']  # [4, 32, 1, 1, 1]
            model.final_conv[0].weight.copy_(pretrained_w[:out_channels])
        if 'final_conv.0.bias' in state and model.final_conv[0].bias is not None:
            pretrained_b = state['final_conv.0.bias']    # [4]
            model.final_conv[0].bias.copy_(pretrained_b[:out_channels])

    model.eval()
    return model


def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0)
        output_tensor = model(input_tensor)
        output_data = output_tensor.squeeze(0).cpu().numpy()
    return output_data


if __name__ == "__main__":
    input_file_path = "./data/frcmatrix/frc_matrix6382.npy"
    model_file_path = "./model/best_model.pth"
    output_file_paths = [
        "./data/predictdata/output6382_1.npy",
        "./data/predictdata/output6382_2.npy",
        "./data/predictdata/output6382_3.npy",
    ]

    input_data = np.load(input_file_path)
    model = load_model(model_file_path, out_channels=3)

    start_time = time.time()
    output_data = predict(model, input_data)  # (3, D, H, W)
    end_time = time.time()

    for i in range(3):
        np.save(output_file_paths[i], output_data[i])

    print("Prediction completed and output saved.")
    print(f"预测耗时: {end_time - start_time:.4f} 秒")