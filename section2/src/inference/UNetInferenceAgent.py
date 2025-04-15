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

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        # Pad coronal (Y) and sagittal (Z) dimensions to match patch size
        conformant_shape = (volume.shape[0], self.patch_size, self.patch_size)
        volume_padded = med_reshape(volume, new_shape=conformant_shape)

        prediction = self.single_volume_inference(volume_padded)

        # Crop prediction back to original size
        prediction_cropped = prediction[:, :volume.shape[1], :volume.shape[2]]
        return prediction_cropped

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()
        slices = []

        with torch.no_grad():
            for i in range(volume.shape[0]):
                slice_i = volume[i, :, :]  # axial slice
                slice_i = slice_i[None, None, :, :]  # [1, 1, H, W]
                slice_tensor = torch.from_numpy(slice_i).float().to(self.device)
                pred = self.model(slice_tensor)  # [1, num_classes, H, W]
                pred_class = torch.argmax(pred, dim=1).cpu().numpy()  # [1, H, W]
                slices.append(pred_class[0])

        return np.array(slices)
