import torch
from torchvision.datasets import CocoDetection
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
import albumentations as A
from torchvision.transforms import ToTensor


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ann_path, sup_path, transform=None):
        self.coco = CocoDetection(img_path, ann_path)
        self.transform = transform
        self.support_by_cat = dict()
        for cat_id in os.listdir(sup_path):
            cat_folder = os.path.join(sup_path, cat_id)
            files = [os.path.join(cat_folder, filename) for filename in os.listdir(cat_folder)]
            self.support_by_cat[int(cat_id)] = files
    
    def __len__(self):
        return len(self.coco)
    
    def __getitem__(self, idx):
        X, y_list = self.coco[idx]

        X = np.array(X)
        if len(y_list) == 0:
            return None
        y = np.random.choice(y_list) # TODO: iterate over instances?

        cat_id = y['category_id']

        support_file = np.random.choice(self.support_by_cat[cat_id])
        support_image = imread(support_file)

        res = {
            'images': X,
            'bboxes': np.array([x['bbox'] for x in y_list if x['category_id'] == cat_id]),
            'supports': support_image,
        }
        return res

def my_collate(batch):
    res = dict()
    img_h_mean = np.mean([x['images'].shape[0] for x in batch if x], dtype=np.int32)
    img_w_mean = np.mean([x['images'].shape[1] for x in batch if x], dtype=np.int32)
    supp_h_mean = np.mean([x['supports'].shape[0] for x in batch if x], dtype=np.int32)
    supp_w_mean = np.mean([x['supports'].shape[1] for x in batch if x], dtype=np.int32)
    images = []
    supports = []
    bbox_points = []
    for item in batch:
        if item is None:
            continue
        item['images'] = resize(item['images'], (img_h_mean, img_w_mean))
        item['supports'] = resize(item['supports'], (supp_h_mean, supp_w_mean))
        images.append(ToTensor()(item['images']))
        supports.append(ToTensor()(item['supports']))
        bbox_points.append(torch.tensor(item['bboxes']))

    res['images'] = torch.stack(images)
    res['supports'] = torch.stack(supports)
    res['bboxes'] = bbox_points
    return res

def get_dataloader(img_path, ann_path, sup_path, batch_size, transform=None):
    ds = CocoDataset(img_path, ann_path, sup_path, transform=transform)
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=my_collate, shuffle=True)
    return ds_loader
    

if __name__ == '__main__':
    ds_loader = get_dataloader('val2017', 'annotations/instances_val2017.json', 'val2017_supp', batch_size=32)
    for idx, item in enumerate(ds_loader):
        print(item['images'].shape, item['supports'].shape, len(item['bboxes']))
        if idx == 3:
            break