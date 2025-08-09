"""Load model saved to TorchScript and run inference with an example image."""

import os
from math import isclose

import numpy as np
import torch
from visualization import gen_animation


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
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

    np_data = np.fromfile("data/input_tensor.dat", dtype=precision)
    np_data = np_data.reshape(transposed_shape)
    np_data = np_data.transpose()
    input_tensor = torch.from_numpy(np_data)

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        output = model.forward(input_tensor)

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = input_tensor.to(torch.device("cuda"))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    input_arr = input_tensor.squeeze().numpy()
    out_arr = output.squeeze().detach().numpy()

    return input_arr, out_arr



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--filepath",
        help="Path to the file containing the PyTorch model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parsed_args = parser.parse_args()
    filepath = parsed_args.filepath
    saved_model_file = os.path.join(filepath, "saved_model_cuda.pt")

    device_to_run = "cuda"

    batch_size_to_run = 1

    infer_input, infer_output = deploy(saved_model_file, device_to_run, batch_size_to_run)

    if not os.path.exists("./results"):
            os.mkdir("results")
        
    gen_animation(np.concat([infer_input, infer_output], axis=0), save_path="./results/python_result.gif")

