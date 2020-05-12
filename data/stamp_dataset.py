### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from data.segmentation_dataset import SegmentationDataset

class StampDataset(SegmentationDataset):
    def initialize(self, opt):
        super(StampDataset, self).initialize(opt)
        self.class_of_interest = [30, 31, 32, 33]  # will define it in child

    def name(self):
        return 'Stamp'
