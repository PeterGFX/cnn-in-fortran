"""Load model saved to TorchScript and run inference with an example image."""

import os

import numpy as np
import torch
from visualization import gen_animation


def read_data(batch_size: int = 1) -> np.ndarray:
    """
    Load TorchScript model and run inference with Tensor from example image.

    Parameters
    ----------
    saved_model : str
        location of model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with Tensor of ones
    """
    transposed_shape = [320, 320, 5, batch_size]
    precision = np.float32

    input_data = np.fromfile("data/input_tensor.dat", dtype=precision)
    input_data = input_data.reshape(transposed_shape)
    input_data = input_data.transpose().squeeze()

    output_data = np.fromfile("data/output_fortran.dat", dtype=precision)
    output_data = output_data.reshape(transposed_shape)
    output_data = output_data.transpose().squeeze()
    
    return input_data, output_data



if __name__ == "__main__":
    
    batch_size_to_run = 1

    infer_input, infer_output = read_data(batch_size_to_run)

    if not os.path.exists("./results"):
            os.mkdir("results")
        
    gen_animation(np.concat([infer_input, infer_output], axis=0), save_path="./results/fortran_result.gif")

