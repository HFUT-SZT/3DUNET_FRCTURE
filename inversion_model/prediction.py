import torch
import numpy as np                          
from model import UNet3D  # Import your model architecture

# Load the trained model
model = UNet3D()

# # Load the model weights from the saved .pth file
# model.load_state_dict(torch.load('../DL0/model/best_model3T.pth'))
                                
# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model.load_state_dict(
    torch.load(
        './model/best_model.pth',
        map_location=device,
        weights_only=True
    )
)
model.to(device)
model.eval()


input_file_1 = './data/interdata/inter_normalized_pressure6580_1.npy'
input_file_2 = './data/interdata/inter_normalized_pressure6580_2.npy'
input_file_3 = './data/interdata/inter_normalized_pressure6580_3.npy'
input_file_4 = './data/interdata/inter_normalized_pressure6580_4.npy'
 
# Load the 3D matricespp
input_matrix_1 = np.load(input_file_1)
input_matrix_2 = np.load(input_file_2)
input_matrix_3 = np.load(input_file_3)
input_matrix_4 = np.load(input_file_4)

# Combine the matrices into a tensor of shape (1, 4, 128, 64, 32)
input_data = np.stack([input_matrix_1, input_matrix_2, input_matrix_3,input_matrix_4], axis=0)
input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Move the input data to the same device as the model (GPU or CPU)
input_data = input_data.to(device)

# Perform the prediction
with torch.no_grad():
    prediction = model(input_data)

# The prediction is of shape (1, 1, 128, 64, 32), squeeze to get the 3D matrix
predicted_output = prediction.squeeze(0).squeeze(0)  # Shape will be (128, 64, 32)

# Convert the predicted tensor to a NumPy array if neededs
predicted_output_numpy = predicted_output.cpu().numpy()

# Save the prediction as a .npy file
# np.save('./predictdata/643noise_predicted_test_output490.npy', predicted_output_numpy)
np.save('./data/predictdata/6580predicted_test_output_EC.npy', predicted_output_numpy)
print("Prediction complete. Output saved as 'predicted_output.npy'.")
