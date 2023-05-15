import os
import numpy as np
import math
import cv2
import random

def calc_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20*math.log10(PIXEL_MAX/math.sqrt(mse))

def add_gaussian_noise(img, sigma):
    noisy_image = np.zeros_like(img, dtype=np.float32)
    gaussian_noise = np.random.normal(0, sigma, (img.shape[0], img.shape[1]))
    
    if len(img.shape) == 2:
        noisy_image = img + gaussian_noise    
    else:
        for i in range(img.shape[2]):
            noisy_image[:, :, i] = img[:, :, i] + gaussian_noise
            
    noisy_image = np.clip(noisy_image, a_min=0.0, a_max=255.0)
    return noisy_image

def synthetic_blur(img_path, ksize=25, std=1.6, sigma=2, add_noise=True):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=std)
            
    if add_noise:
        blur_img = add_gaussian_noise(blur_img, sigma=sigma)
    return blur_img

def get_2d_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum

class EstimateBlur:
    def __init__(self, hq_filelist, lq_filelist, output_root_dir):
        self.hq_filelist = hq_filelist
        self.lq_filelist = lq_filelist
        self.output_root_dir = output_root_dir
        self.hq_dir = os.path.join(output_root_dir, 'EM/HQ')
        self.lq_dir = os.path.join(output_root_dir, 'EM/LQ/0')
        
        os.makedirs(self.hq_dir, exist_ok=True)
        os.makedirs(self.lq_dir, exist_ok=True)
         
    def create_synthetic_paired_dataset(self):
        print('Creating Synthetic dataset...')
        ksize, sigma = self.return_best_config()
        i = 0
        for hq_file in self.hq_filelist:
            img = cv2.imread(hq_file, cv2.IMREAD_GRAYSCALE)
            img = 255 - img
            blur_img = synthetic_blur(hq_file, ksize=ksize, sigma=sigma)
            fname = '{:0>4}.png'.format(i)
            hq_fpath = os.path.join(self.hq_dir, fname)
            blur_fpath = os.path.join(self.lq_dir, fname)
            cv2.imwrite(hq_fpath, img)
            cv2.imwrite(blur_fpath, blur_img)
            i += 1
        
        i = 0
        for lq_file in self.lq_filelist:
            img = cv2.imread(lq_file, cv2.IMREAD_GRAYSCALE)
            img = 255 - img
            fname = '{:0>4}.png'.format(i)
            lq_fpath = os.path.join(self.output_root_dir, 'out_of_focus', fname)
            cv2.imwrite(lq_fpath, img)
            i += 1
        
        print('Finished creating synthetic dataset')
            
        
    def return_best_config(self):
        ksizes = list(range(5, 26, 2))
        sigmas = list(range(1, 26))
        psnrs = []
        num_samples = int(min(len(self.hq_filelist), len(self.lq_filelist))/2)
        hq_samples = random.sample(self.hq_filelist, num_samples)
        lq_samples = random.sample(self.lq_filelist, num_samples)
        
        print('Estimating best configuration')
        for ksize in ksizes:
            for sigma in sigmas:
                total_psnr = 0.0
                i = 0
                for hq_sample in hq_samples:
                    # print('Kernel Size: {}, Sigma: {}'.format(ksize, sigma))
                    blur_img = synthetic_blur(hq_sample, ksize=ksize, sigma=sigma)
                    blur_fft = get_2d_fourier(blur_img)
                    for lq_sample in lq_samples:
                        lq_img = cv2.imread(lq_sample, cv2.IMREAD_GRAYSCALE)
                        lq_fft = get_2d_fourier(lq_img)
                        psnr = calc_psnr(lq_fft, blur_fft)
                        total_psnr += psnr
                        i += 1
                avg_psnr = total_psnr/i
                config_dict = {
                    'ksize': ksize,
                    'sigma': sigma,
                    'psnr': avg_psnr
                }
                psnrs.append(config_dict)
                
        psnrs = sorted(psnrs, key=lambda x : x['psnr'], reverse=True)
        return psnrs[0]['ksize'], psnrs[0]['sigma']
    