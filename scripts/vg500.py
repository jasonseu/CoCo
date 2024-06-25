# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2024-3-27
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2024 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import json
from tqdm import tqdm
from collections import Counter

data_dir = 'datasets/VisualGenome'
image_dir1 = os.path.join(data_dir, 'VG_100K')
image_dir2 = os.path.join(data_dir, 'VG_100K_2')
partition_dir = os.path.join(data_dir, 'ssgrl_partition')
save_dir = 'data/vg500'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(partition_dir, 'train_list_500.txt'), 'r') as fr:
    ssgrl_train_list = [temp.strip() for temp in fr]
with open(os.path.join(partition_dir, 'test_list_500.txt'), 'r') as fr:
    ssgrl_test_list = [temp.strip() for temp in fr]

objId_imgName = {}
with open(os.path.join(partition_dir, 'vg_category_500_labels_index.json'), 'r') as fr:
    for img_name, object_ids in json.load(fr).items():
        objId_imgName[img_name] = object_ids
            

train_data = []
for img_name in ssgrl_train_list:
    img_labels = objId_imgName[img_name]
    img_path = os.path.join(image_dir1, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir2, img_name)
        if not os.path.exists(img_path):
            raise Exception('file {} not found!'.format(img_path))
    train_data.append('{}\t{}\n'.format(img_path, ','.join(map(str, img_labels))))

test_data = []
for img_name in ssgrl_test_list:
    img_labels = objId_imgName[img_name]
    img_path = os.path.join(image_dir1, img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(image_dir2, img_name)
        if not os.path.exists(img_path):
            raise Exception('file {} not found!'.format(img_path))
    test_data.append('{}\t{}\n'.format(img_path, ','.join(map(str, img_labels))))
    
with open(os.path.join(save_dir, 'train.txt'), 'w') as fw:
    fw.writelines(train_data)
with open(os.path.join(save_dir, 'test.txt'), 'w') as fw:
    fw.writelines(test_data)
with open(os.path.join(save_dir, 'label.txt'), 'w') as fw:
    fw.writelines(['{}\n'.format(t) for t in range(500)])
