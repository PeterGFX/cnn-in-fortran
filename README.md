# cnn-in-fortran
 HPC4WC group project about CNN performance in HPC

List of contents:

### /src/train_and_test_unet:
##### Code for training and validation the UNet temperature field extrapolation model.
dataset.py: Creates a iteratable pytorch dataset from temperature field data.\n
dataloader.py: Class for dataloader for both training and validation.\n
models.py: Implementation of the UNet model.\n
train_and_test.py: Main training and validation loop.\n
training_record_plot & visulization.py: Plotting scripts for results and training statistics.\n
pt2ts.py: Converts and save pytorch model to torchscript.\n
