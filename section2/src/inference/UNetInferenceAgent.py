"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet
from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):
        self.model = model if model is not None else UNet(num_classes=3)
        self.patch_size = patch_size
        self.device = device

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

    def single_volume_inference_unpadded(self, volume):
        """
        Pads a volume to patch size, runs inference, then crops prediction to original size.
        """
        conformant_shape = (volume.shape[0], self.patch_size, self.patch_size)
        volume_padded = med_reshape(volume, new_shape=conformant_shape)

        prediction = self.single_volume_inference(volume_padded)

        # Crop prediction back to original size
        prediction_cropped = prediction[:, :volume.shape[1], :volume.shape[2]]
        return prediction_cropped

    def single_volume_inference(self, volume):
        """
        Runs inference on a single 3D volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array [slices, height, width]

        Returns:
            3D NumPy array with predicted segmentation masks
        """
        slices = np.zeros(volume.shape, dtype=np.uint8)

        with torch.no_grad():
            for ix in range(volume.shape[0]):
                # Normalize individual slice to [0, 1]
                slc = volume[ix, :, :].astype(np.float32)
                slc = slc / np.max(slc) if np.max(slc) > 0 else slc

                # Prepare tensor shape: [1, 1, H, W]
                slc_tensor = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).to(self.device)

                # Forward pass
                pred = self.model(slc_tensor)  # [1, num_classes, H, W]
                pred_class = torch.argmax(pred, dim=1).cpu().numpy()  # [1, H, W]

                # Store result
                slices[ix, :, :] = pred_class[0]

        return slices
