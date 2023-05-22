
# DeblurPipeline

Repository to help create a dataset compatible for training and testing on GRAN_GAN

  

# How to use

To run this code on your data you need a text file containing a list of the complete file paths for your microscopy section. The code assumes that all these files belong to the same section

  

## Metrics

The repository provides three metrics for calculating image quality:

 1. `BRISQUE`
 2. `Variance Based`
 3. `Sharpness`

The source code for which can be found in `utils/metrics.py` 
You can also provide a threshold for deciding the quality of the images between good and bad

## Synthetic Dataset
The pipeline generates a synthetic dataset with HQ and LQ folders. Provide a root output directory with `--output_root_dir` and in that `EM/HQ` contains the high quality images and `EM/LQ/0` contains synthetically generated low quality images

## Command

    python  main.py  --filepaths  '../file_list.txt'  --threshold  100  --output_root_dir  '../synthetic_dataset/0025'
