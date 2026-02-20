The repository holds the code and models described in the manuscript "Swamp-Eye: A deep learning model for monitoring wetlands change across the globe" by Andros et al 2026.

This is a deep learning tool designed for the automated segmentation of swamp-land using Sentinel-2 multispectral imagery. The optimal model is the ResUNet34 architecture using the focal_dice loss function and the model expects a 6-band GeoTIFF stack. Additional details related to the model architecture are detailed in the manuscript.

An example script for using the model is contained in the repository and is called "Example.ipynb". The other notebook called "trainer.ipynb" is included for reproducibility, in case a user wishes to review the code that generated and trained the models described in the manuscript.

⚠️ Important: Currently, there is a known issue when downloading this repository as a ZIP file. GitHub LFS (Large File Storage) may replace the model weights (.h5py) with a small pointer file, causing the model to fail to load.
While a more permanent fix in under development, if you use the ZIP download please go to the model/pretrained/ folder on GitHub and download the .h5py file individually by clicking the "Download" button on that specific file and then placing them in the model weights folder (model/pretrained/model_architecture.hpy5).
