The repository holds the code and models described in the manuscript "Swamp-Eye: A deep learning model for monitoring wetlands change across the globe" by Andros et al 2026.

This is a deep learning tool designed for the automated segmentation of swamp-land using Sentinel-2 multispectral imagery. 

An example script for downloading imagery from Google Earth Engine is contained in the file "GEE Download Script.txt".

An example script for using the model is contained in the repository and is called "Example.ipynb". The other notebook called "trainer.ipynb" is included for reproducibility, in case a user wishes to review the code that generated and trained the models described in the manuscript.

⚠️ Important: Currently, there is a known issue when downloading this repository as a ZIP file. GitHub LFS (Large File Storage) may replace the model weights (.h5py) with a small pointer file, causing the model to fail to load.
While a more permanent fix in under development, if you use the ZIP download please go to the model/pretrained/ folder on GitHub and download the .h5py file individually by clicking the "Download" button on that specific file and then placing them in the model weights folder (model/pretrained/"insert model architecture here".hpy5).

If you use this repository in your research, please cite the following paper:
Andros, C.S., Conery, I.W., Alvarado, T.R. et al. Swamp-Eye: a deep learning model for monitoring wetlands change across the globe. Sci Rep (2026). https://doi.org/10.1038/s41598-026-39257-1
