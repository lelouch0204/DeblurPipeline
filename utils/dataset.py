import os
from typing import Any
from utils.metrics import MetricFactory
from utils.estimate import EstimateBlur

class Dataset:
    def __init__(self, args):
        self.filepaths = args.filepaths
        self.metric_name = args.metric
        self.output_file = args.output_file
        self.output_root_dir = args.output_root_dir
        
        self.threshold = args.threshold
        
        with open(self.filepaths, 'r') as f:
            filelist = [line.strip() for line in f.readlines()]
        
        self.filelist = filelist
        # print(filelist)
        self.metric = MetricFactory.create_metric(filelist=self.filelist, metric=self.metric_name)
        
    def create_file(self):
        hq_filelist = []
        lq_filelist = []
        with open(self.output_file, "w") as f:
            for quality_info in self.quality_list:
                result = None
                if self.metric_name.lower() == 'brisque':
                    result = 'HQ' if quality_info['Score'] <= self.threshold else 'LQ'
                else:
                    result = "LQ" if quality_info["Score"] < self.threshold else "HQ"
                write_str = quality_info['Image Path'] + "," + str(quality_info['Score']) + "," + result + "\n"
                hq_filelist.append(quality_info['Image Path']) if result is 'HQ' else lq_filelist.append(quality_info['Image Path'])
                f.write(write_str) 
        print('Found {} HQ Images and {} LQ Images'.format(len(hq_filelist), len(lq_filelist)))
        return hq_filelist, lq_filelist
    
    def __call__(self):
        print('Starting quality calculation')
        self.quality_list = self.metric.calculate_quality()
        # print('Finished Calculating quality')
        self.hq_filelist, self.lq_filelist = self.create_file()
        self.estimate_blur = EstimateBlur(self.hq_filelist, self.lq_filelist, self.output_root_dir)
        self.estimate_blur.create_synthetic_paired_dataset()