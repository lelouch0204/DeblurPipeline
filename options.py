import argparse

parser = argparse.ArgumentParser(description='Create Dataset and file for deblurring')

parser.add_argument('--filelist', required=True, type=str, help='A txt file containing a list of the images that need to be included in the dataset')
parser.add_argument('--metric', default='brisque', type=str, choices=['brisque', 'variance', 'sharpness'], help='Metric on which the image quality is to be judged (default: brisque)')
parser.add_argument('--output_file', default='image_quality.txt', type=str, help='Path of the txt file in which you want the image quality output results')
parser.add_argument('--threshold', required=True, type=int, help='Threshold to decide the score for classifying images as good and bad')
parser.add_argument('--output_root_dir', required=True, type=str, help='Directory in which you want your synthetically generated dataset to be placed') 

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
