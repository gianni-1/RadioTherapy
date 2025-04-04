import argparse
import torch
import nibabel as nib
import numpy as np

# Dummy dose prediction network; replace with the actual model architecture.
class DoseNet(torch.nn.Module):
    def __init__(self):
        super(DoseNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(16, 1, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    # If the checkpoint holds a dictionary with 'dose_net', use that.
    #if isinstance(checkpoint, dict) and "dose_net" in checkpoint:
    #    state_dict = checkpoint["dose_net"]
    #else:
    state_dict = checkpoint
    model = DoseNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_dose(input_filepath, output_filepath, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load patient CT scan (Nifti)
    img = nib.load(input_filepath)
    data = img.get_fdata()
    # Normalize data to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    
    # Prepare tensor with shape (1, 1, depth, height, width)
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Load trained dose prediction model
    model = load_model(model_path, device)
    
    # Run inference
    with torch.no_grad():
        dose_prediction = model(tensor_data)
    
    # Convert prediction to numpy array and create a Nifti image (preserving input header/affine)
    dose_np = dose_prediction.squeeze().cpu().numpy()
    dose_img = nib.Nifti1Image(dose_np, affine=img.affine, header=img.header)
    nib.save(dose_img, output_filepath)
    print("Dose distribution saved to:", output_filepath)
    # Return dose distribution for visualization
    return dose_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict dose distribution for patient CT scan using ML.")
    parser.add_argument("input", help="Path to patient's CT scan (Nifti file)")
    parser.add_argument("model", help="Path to trained DoseNet model checkpoint")
    parser.add_argument("output", help="Path to save the predicted dose distribution (Nifti file)")
    args = parser.parse_args()
    
    predict_dose(args.input, args.output, args.model)