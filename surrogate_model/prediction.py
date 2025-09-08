import torch
import numpy as np
import time
from model import UNet3D


def load_model(model_path, out_channels=3):
    model = UNet3D(out_channels=out_channels)
    state = torch.load(model_path, map_location=torch.device('cpu'))

    filtered_state = {k: v for k, v in state.items() if not k.startswith('final_conv.0.')}
    _load_result = model.load_state_dict(filtered_state, strict=False)

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
    input_file_path = "./data/frcmatrix/frc_matrix6400.npy"
    model_file_path = "./model/best_model.pth"
    output_file_paths = [
        "./data/predictdata/output6400_1.npy",
        "./data/predictdata/output6400_2.npy",
        "./data/predictdata/output6400_3.npy",
    ]

    input_data = np.load(input_file_path)
    model = load_model(model_file_path, out_channels=3)

    start_time = time.time()
    output_data = predict(model, input_data)  # (3, D, H, W)
    end_time = time.time()

    for i in range(3):
        np.save(output_file_paths[i], output_data[i])

    print("Prediction completed and output saved.")
    print(f"time cost: {end_time - start_time:.4f} s")