import numpy as np
import torch
from torch import nn


class PINNMEOutput:

    def __init__(self, model_path, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(model_path, map_location=self.device)

        self.parameter_model = state['parameter_model']
        self.parameter_model = nn.DataParallel(self.parameter_model)
        self.parameter_model.eval()

        self.cube_shape = state['cube_shape']
        self.lambda_grid = state['lambda_grid']
        self.data_range = state['data_range']

    @torch.no_grad()
    def load(self, coords, batch_size=4096):
        coords_shape = coords.shape
        coords_tensor = torch.tensor(coords, dtype=torch.float32).reshape(-1, coords.shape[-1])

        parameters = {}

        n_batches = int(np.ceil(coords_tensor.shape[0] / batch_size))
        for i in range(n_batches):
            batch = coords_tensor[i * batch_size:(i + 1) * batch_size].to(self.device)
            pred = self.parameter_model(batch)
            for key, value in pred.items():
                if key not in parameters:
                    parameters[key] = []
                value = value.cpu().numpy()
                parameters[key].append(value)

        parameters = {key: np.concatenate(value).reshape(*coords_shape[:-1], -1)
                      for key, value in parameters.items()}

        return parameters

    def load_cube(self):
        coords = np.meshgrid(
            np.linspace(self.data_range[0][0], self.data_range[0][1], self.cube_shape[0], dtype=np.float32),
            np.linspace(self.data_range[1][0], self.data_range[1][1], self.cube_shape[1], dtype=np.float32),
            np.linspace(self.data_range[2][0], self.data_range[2][1], self.cube_shape[2], dtype=np.float32),
            indexing='ij')
        coords = np.stack(coords, axis=-1)

        return self.load(coords)

    def load_time(self, time):
        coords = np.meshgrid(
            np.linspace(self.data_range[0][0], self.data_range[0][1], self.cube_shape[0], dtype=np.float32),
            np.linspace(self.data_range[1][0], self.data_range[1][1], self.cube_shape[1], dtype=np.float32),
            np.ones(1, dtype=np.float32) * time,
            indexing='ij')
        coords = np.stack(coords, axis=-1)

        return self.load(coords)


def to_cartesian(b_field, theta, chi):
    b_x = b_field * np.sin(theta) * np.cos(chi)
    b_y = b_field * np.sin(theta) * np.sin(chi)
    b_z = b_field * np.cos(theta)
    return np.stack([b_x, b_y, b_z], axis=-1)

def to_spherical(b):
    b_field = np.linalg.norm(b, axis=-1)
    theta = np.arccos(b[..., 2] / b_field)
    chi = np.arctan2(b[..., 1], b[..., 0])
    return b_field, theta, chi