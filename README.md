# Deep Learning for Wood Moisture Monitoring and Deformation Analysis Using 4D-CT

This repo is the official implementation of ‘Cutting-Edge Deep Learning and 4D-CT Imaging: Revolutionizing Moisture Monitoring and Deformation Analysis in Wood during Drying’. It currently includes code for the following tasks:

## 1. Abstract
Wood plays a pivotal role in building a sustainable future due to its renewable properties, carbon sequestration capabilities, and versatility across applications such as construction, furniture, and energy production. However, the wood drying process poses substantial challenges. To address these, our study introduces a novel method combining U-net deep learning image registration with Four-Dimensional Computed Tomography (4D-CT). This approach enables time-resolved, three-dimensional visualization of moisture migration and deformation during the wood drying process. 

Our methodology reveals detailed information about wood's anisotropic nature, supporting advanced modeling and optimization of drying schedules, ultimately improving wood quality.

## 2. Getting Started

### 2.1 Preprocessing
The raw CT files are normalized and rigidly registered with the allometric quantities after undergoing filtering processes. Preprocessing is crucial for improving the quality of model training.

- **readCTslices.m**: Reads all DICOM files in a folder and reorganizes them into a matrix.
- **normalizeCTData.m**: Normalizes CT data and removes background noise.
- **registerImages.m**: Aligns CT matrix data at translation and rotation levels.
- **nomorliaztion.m**: Performs batch normalization.
- **nii_save.m**: Batch aligns data and stores it as NIfTI (.nii) files.

### 2.2 Deep-learning Model
The VoxelMorph library (https://github.com/voxelmorph/voxelmorph) is utilized as the foundation for deep learning-based registration, with significant improvements to the generator, mesh structure, and loss functions for wood-specific applications.

- **train.py**: Main program for training the deep learning model. You can customize mesh structures, loss functions, and other parameters. During each epoch, images are outputted for alignment evaluation.
- **test.py**: Inference script.
- **visualise.py**: Functions for visualizing results.
