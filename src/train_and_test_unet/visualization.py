import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from models import UNet
from dataloader import get_dataloaders

import os
import torch

def gen_animation(seq, save_path):

    print("generating animation...", flush=True)
    fig = plt.figure()
    im = plt.imshow(seq[0,...], animated=True)

    def updatefig(frame):

        if (frame<seq.shape[0]-1):
            frame += 1
        else:
            frame=0
        im.set_array(seq[frame,...])
        return im,

    ani = animation.FuncAnimation(fig, updatefig,  blit=True, interval=500, frames=seq.shape[0])
    writer = animation.PillowWriter(fps=1,
                                bitrate=1000)
    ani.save(save_path, writer=writer)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_len=5
    out_len=5
    
    with torch.no_grad():

        model = UNet(in_channels=in_len, 
                 num_classes=out_len, 
                 depth=5, merge_mode='concat').to(device)
    
        model.load_state_dict(torch.load('./checkpoints/unet_v1.pt', weights_only=True))
        model.eval()

        test_loader, _ = get_dataloaders(1,
                                    in_len=in_len,
                                    out_len=out_len,
                                    )

        test_input, test_target = next(iter(test_loader))
        test_input, test_target = test_input.to(device), test_target.to(device).float()

        test_output = model(test_input)

        test_input = test_input.squeeze().cpu().numpy()
        test_target = test_target.squeeze().cpu().numpy()
        test_output = test_output.squeeze().cpu().numpy()

        if not os.path.exists("./results"):
            os.mkdir("results")

        print(test_input.shape)
        print(test_target.shape)
        print(test_output.shape)
        
        gen_animation(np.concat([test_input, test_output], axis=0), save_path="./results/output.gif")
        gen_animation(np.concat([test_input, test_target], axis=0), save_path="./results/ground_truth.gif")

