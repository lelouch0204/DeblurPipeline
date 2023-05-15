import cv2
import numpy as np
import brisque
from tqdm import tqdm

class ImageQuality:
    def __init__(self, filelist):
        self.filelist = filelist
        
    def calculate_quality(self):
        pass

class BRISQUE(ImageQuality):
    def calculate_quality(self):
        quality_list = []
        obj = brisque.BRISQUE(url=False)
        for file in tqdm(self.filelist):
            print(file)
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = 255 - img
            # print(img.shape)
            score = obj.score(img)
            info_dict = {
                'Image Path': file,
                'Score': score
            }
            quality_list.append(info_dict)
        print('Finished calculating image quality')
        return quality_list
    
class VarianceBased(ImageQuality):
    def calculate_image_quality(file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = 255 - img
        # img_inverted = 255 - img

        xs = img.shape[1]
        ys = img.shape[0]

        ##IMAGE QUALITY DEFINITIONS
        dsamp = 200
        sig_thresh = 0.01
        top = 0.5

        yshift = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        xshift = [0, 1, 2, 0, 1, 2, 0, 1, 2]

        sur = np.array([[2, 4, 6],
                        [1, 2, 3],
                        [4, 8, 6], 
                        [4, 5, 6],
                        [1, 5, 7],
                        [1, 4, 7],
                        [2, 6, 8],
                        [2, 5, 8]])

        sur = sur - 1

        cent = np.array([[1, 5, 3],
                            [4, 5, 6],
                            [7, 5, 9],
                            [7, 8, 9],
                            [2, 4, 8],
                            [2, 5, 8],
                            [3, 5, 9],
                            [3, 6, 9]])

        cent = cent - 1

        dsampy = int(np.fix((ys-5)/dsamp))
        dsampx = int(np.fix((xs-5)/dsamp))
        # print('dsampy: {}, dsampx: {}'.format(dsampy, dsampx))


        id_zeros = np.zeros(shape=(dsamp, dsamp, len(yshift)))
        # print('Image size before cropping: {}'.format(img.shape))
            
        for j in range(len(yshift)):
            region_of_interest = [
                                [3 + xshift[j] * dsampx, xs],
                                [3 + yshift[j] * dsampx, ys]
                            ]
            # print(ys_new, xs_new)
            is_temp = img[3+yshift[j]:ys:dsampy, 3+xshift[j]:xs:dsampx]
            # print(i_s.shape)
            # print(source.shape)
            i_s = is_temp[:dsamp, :dsamp]
            id_zeros[:, :, j] = 255 - i_s
            
        max_id = np.mean(id_zeros, axis=2)
        dev_i = np.std(id_zeros, axis=2)
        last_values = id_zeros[-1, :, :]

        sats = np.logical_or(id_zeros == 255, id_zeros <= 1).astype(int)
        percent_sat = np.sum(sats)/sats.size*100
        sum_sats = np.sum(sats, axis=2)
        use_vals = dev_i[sum_sats<1]

        region_quality = None
        if use_vals.size > 5:
            sort_mean = np.sort(use_vals)
            # print(int(-(1 - np.fix(sort_mean.size*sig_thresh))))
            thresh_val = sort_mean[int(-(1 + np.fix(sort_mean.size*sig_thresh)))]
            temp = np.logical_and(dev_i>=thresh_val, sum_sats==0)
            use_sig = np.argwhere(np.logical_and(dev_i>=thresh_val, sum_sats==0))
            # print('Shape of use_sig: {}'.format(use_sig.shape))
            show_used = 255 - max_id
            show_used[temp] = 1000
            
            difs = np.empty([id_zeros.shape[0], id_zeros.shape[1], len(cent)])
            use_difs = [None]*len(cent)
            for f in range(len(cent)):
                dif = np.mean(id_zeros[:, :, cent[f]], axis=2) - np.mean(id_zeros[:, :, sur[f]], axis=2)
                # print('Shape of dif: {}'.format(dif.shape))
                difs[:, :, f] = dif
                dif_masked = dif[temp]
                use_difs[f] = dif_masked
            
            # print('use_difs:')
            # print(use_difs[0].shape)
                
            samp_size = max(use_sig.shape)
            groups = np.array([[0, 2], [1, 3], [4, 6], [5, 7]])
            top_con = np.zeros(shape=(len(groups)))
            for f in range(len(groups)):
                group_indices = groups[f]
                group_arrays = [use_difs[i] for i in group_indices]
                vals = np.abs(np.concatenate(group_arrays))
                # print('vals shape: {}'.format(vals.shape))
                sorted_vals = np.sort(vals)
                thresh = sorted_vals[- 1 - round(len(sorted_vals)*top)]
                top_con[f] = np.mean(vals[vals>=thresh])
                
            # print('top_con: {}'.format(top_con))
            mean_vals = np.zeros((samp_size, len(groups)))
            max_vals = np.zeros((samp_size, len(groups)))
            # print(mean_vals.shape)
            
            for f in range(len(groups)):
                group_indices = groups[f]
                group_arrays = [use_difs[i] for i in group_indices]
                vals = np.stack(group_arrays, axis=1)
                mean_vals[:, f] = np.abs(np.mean(vals, axis=1))       
                
            # max_vals[:, 0] = np.max([mean_vals[:, 0], mean_vals[:, 2]], axis=0)
            max_vals[:,0] = np.amax(mean_vals[:,[0,2]], axis=1).reshape(-1)
            

            
            top_max = np.zeros((2))
            for f in range(2):
                max_vals = np.amax(mean_vals[:,[f,f+2]], axis=1).reshape(-1)
                sort_vals = np.sort(max_vals)
                thresh = sort_vals[len(sort_vals) - 1 - round(len(sort_vals)*top)]
                top_max[f] = np.mean(max_vals[max_vals>=thresh])
                
            glob_arr = np.mean(id_zeros, axis=2)
            
            glob_vals = np.sort(glob_arr)   
            back_ground = np.median(glob_vals[:dsamp])
            top_glob_m = np.mean(glob_arr[temp])
            value_range = top_glob_m - back_ground
            
            vert_qual = ((top_con[1]/top_con[0])-1) * 100
            horz_qual = ((top_con[3]/top_con[2])-1) * 100
            region_quality = ((top_max[1]/top_max[0])-1) * 100  
            # print('Vert Qual: {}, Horz Qual: {}'.format(vert_qual, horz_qual))
            
            
        else:
            region_quality = -100

        return region_quality
    
    def calculate_quality(self):
        quality_list = []
        for file in tqdm(self.filelist):
            score = self.calculate_image_quality(file)
            info_dict = {
                'Image Path': file,
                'Score': score
            }
            quality_list.append(info_dict)
        return quality_list
    
class Sharpness(ImageQuality):
    def tenengrad(img, step=1, ksize=5):
        Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=step, dy=0, ksize=ksize)
        Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=step, ksize=ksize)
        FM = Gx**2 + Gy**2
        return 6000*cv2.mean(FM)[0]/(img.shape[0]*img.shape[1])

    def calculate_image_quality(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = 255 - img
        sharpness = self.tenengrad(img, 1, 3)
        brightness = np.mean(img)
        contrast = np.std(img)
        h = np.histogram(img, 255, range=[0, 256])
        clip_black = 100*np.sum(h[0][:5])/(img.shape[0]*img.shape[1])
        clip_white = 100*np.sum(h[0][-5:])/(img.shape[0]*img.shape[1])
        
        return [sharpness, brightness, contrast, clip_black, clip_white] 
    
    def calculate_quality(self):
        quality_list = []
        for file in tqdm(self.filelist):
            score = self.calculate_image_quality(file)
            info_dict = {
                'Image Path': file,
                'Score': score
            }
            quality_list.append(info_dict)
        return quality_list
    
class MetricFactory:
    def create_metric(filelist, metric='BRISQUE'):
        print(filelist)
        if metric.lower() == 'brisque':
            return BRISQUE(filelist)
        elif metric.lower() == 'variance':
            return VarianceBased(filelist)
        elif metric.lower() == 'sharpness':
            return Sharpness(filelist)
        else:
            print('Please select one of the following options: [BRISQUE | Variance | Sharpness]')
            return None