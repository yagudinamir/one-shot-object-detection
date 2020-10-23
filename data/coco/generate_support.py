from torchvision.datasets import CocoDetection
from skimage.io import imsave
import os
import numpy as np
import pandas as pd
import tqdm

def generate_support(ann_path, img_path, min_area=5):
    dataset = CocoDetection(img_path, ann_path)
    for X, y_list in tqdm.tqdm(dataset):
        X = np.array(X)
        for y in y_list:
            x1, y1, dx, dy = map(int, y['bbox'])
            if dx * dy < min_area:
                continue
            inst = X[y1:y1+dy, x1:x1+dx]
            cat_id, inst_id = y['category_id'], y['id']
            cat_dir = f'{img_path}_supp/{cat_id}'
            os.makedirs(cat_dir, exist_ok=True)
            inst_path = os.path.join(cat_dir, str(inst_id) + '.jpg')
            imsave(inst_path, inst)


if __name__ == '__main__':
    ann_path_val = 'annotations/instances_val2017.json'
    img_path_val = 'val2017'
    ann_path_train = 'annotations/instances_train2017.json'
    img_path_train = 'train2017'
    #generate_support(ann_path_val, img_path_val)
    #print("Done with val")
    generate_support(ann_path_train, img_path_train)