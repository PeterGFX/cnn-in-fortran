import numpy as np
import matplotlib.pyplot as plt

arr = np.load('./results/unet_v3_gpu_TS.npy')
print(arr.shape)
plt.plot(arr)
plt.yscale('log')
plt.title("Validation loss")
plt.show() 