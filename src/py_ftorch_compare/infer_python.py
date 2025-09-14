"""Load model saved to TorchScript and run inference with an example image."""

import os
from math import isclose
import time

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
    time_elapsed = 0.

    np_data = np.fromfile("data/input_tensor.dat", dtype=precision)
    np_data = np_data.reshape(transposed_shape)
    np_data = np_data.transpose()
    input_tensor = torch.from_numpy(np_data)
    n_times = 100

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = input_tensor.to(torch.device("cuda"))

        # GPU warmup
        for i in range(10):
            output_gpu = model(input_tensor_gpu)
        torch.cuda.synchronize()
        # count time
        for i in range(n_times):
            start_time = time.time()
            output_gpu = model(input_tensor_gpu)
            torch.cuda.synchronize()
            end_time = time.time()
            print(end_time-start_time)
            time_elapsed += end_time - start_time
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    input_arr = input_tensor.squeeze().numpy()
    out_arr = output.squeeze().detach().numpy()
    print(f"Average inference on {device} took {time_elapsed/n_times} s.")

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
    #filepath = parsed_args.filepath
    #saved_model_file = os.path.join(filepath, "saved_model_cuda.pt")
    saved_model_file = parsed_args.filepath

    device_to_run = "cuda"

    batch_size_to_run = 1

    infer_input, infer_output = deploy(saved_model_file, device_to_run, batch_size_to_run)

    infer_output.tofile("./data/output_python.dat")
    infer_output = infer_output.reshape([1,320,320])

    if not os.path.exists("./results"):
            os.mkdir("results")
        
    gen_animation(np.concat([infer_input, infer_output], axis=0), save_path="./results/python_result.gif")

