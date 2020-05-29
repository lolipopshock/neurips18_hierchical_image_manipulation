import os
import glob
from shutil import copy2
import cv2
from PIL import Image
import json
import numpy as np
import argparse
import shutil 
from skimage import io 
from tqdm import tqdm
from itertools import product

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def copy_file(src, dst):
    if os.path.exists(dst):
        os.rmdir(dst)
    shutil.copytree(src, dst)



def construct_single_class_dataset(image_source_path, anno_source_path, target_paths):
    anno_paths = glob.glob(anno_source_path+'/*.png')
    image_names = list(map(lambda x: os.path.\
                                    basename(x).\
                                    split('_')[0], anno_paths))
    anno_names  = list(map(lambda x: os.path.\
                                    basename(x).\
                                    split('.')[0], anno_paths))
    
    for image_name, anno_path, anno_name in tqdm(zip(image_names[:50], anno_paths, anno_names)):
        
        # Obtain the label and inst
        image_path = os.path.join(image_source_path, image_name+'.jpg')
        label_map  = cv2.cvtColor(cv2.imread(anno_path), cv2.COLOR_BGR2GRAY)
            # label_map contains annotaiton for different parts of a human
        label_mask = label_map>0
            # label_mask is a binary mask where 1 denotes the POI region 
            # and 0 for background

        # Obtain bounding boxes of the mask
        H, W = label_mask.shape
        ys,xs = np.where(label_mask)
        ymin, ymax, xmin, xmax = \
                ys.min(), ys.max(), xs.min(), xs.max()

        # Construct the bounding boxes information
        # 24 and 24000 is only a hard-coded number 
        # Which maps to the 24 pedestrain in the cityscape dataset 
        object_info = {'bbox': [xmin, ymin, xmax, ymax], 'cls':24}
        inst_info = {'imgHeight':H, 
                        'imgWidth':W, 
                        'objects':{str(24000): object_info}}
        
        # Adapt to the original requirements
        cur_label_map = label_mask.astype('uint8') * 24
        cur_inst_map = label_mask.astype('int32') * 24000 

        save_one_sample(target_paths, image_name, anno_name, 
                image_path, cur_inst_map, cur_label_map, inst_info)

            
def save_one_sample(target_paths, image_name, anno_name, 
                image_path, cur_inst_map, cur_label_map, inst_info):

    # Copy image file and rename 
    source = image_path
    target = os.path.join(target_paths['img'], f'{anno_name}.jpg')
    os.symlink(os.path.abspath(source), target) 

    # Save the label and inst map
    target = os.path.join(target_paths['inst'], f'{anno_name}.png')
    Image.fromarray(cur_inst_map).save(target)

    target = os.path.join(target_paths['label'], f'{anno_name}.png')
    Image.fromarray(cur_label_map.astype('uint8')).save(target)
    
    # Save the json bbox file
    target = os.path.join(target_paths['bbox'], f'{anno_name}.json')
    with open(target, 'w') as f:
        json.dump(inst_info, f, cls=NpEncoder)
        print('wrote a bbox summary of %s to %s' % (anno_name, target))

    

parser = argparse.ArgumentParser(description='List the content of a folder')
parser.add_argument('--source_path', default='datasets/LV-MHP-v2',           help='original MHP path', type=str)
parser.add_argument('--target_path', default='datasets/LV-MHP-v2-modified',  help='save path for new MHP dataset', type=str)

# organize image
if __name__ == '__main__':
    
    args = parser.parse_args()
    


    # create the target paths and ensure the paths exist 
    subsets = ['train', 'val']
    target_folders = ['img', 'label', 'inst', 'bbox']
    target_paths = {subset: {name: os.path.join(args.target_path,  f'{subset}_{name}')
                                for name in target_folders}
                                    for subset in subsets}
    
    for _, tpaths in target_paths.items():
        for _, tpah in tpaths.items():
            if not os.path.exists(tpah):
                os.makedirs(tpah)

    # iterate different subsets
    for subset in subsets:
        image_path_source_subset = os.path.join(args.source_path, subset, 'images')
        anno_path_source_subset  = os.path.join(args.source_path, subset, 'parsing_annos')

        construct_single_class_dataset(image_path_source_subset, 
                              anno_path_source_subset, 
                              target_paths[subset])

