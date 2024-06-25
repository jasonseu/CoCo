# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-3-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import argparse
import numpy as np


# generate adjacency matrix
def preprocessing_for_ssgrl(data):
    dir_name = os.path.join('data', data)

    label_path = os.path.join(dir_name, 'label.txt')
    train_path = os.path.join(dir_name, 'train.txt')
    graph_path = os.path.join(dir_name, 'graph.npy')

    categories = [line.strip() for line in open(label_path).readlines()]
    cate2id = {cat:i for i, cat in enumerate(categories)}
    adjacency_matrix = np.zeros((len(categories), len(categories)))

    with open(train_path, 'r') as fr:
        data = [line.strip().split('\t')[1].split(',') for line in fr.readlines()]

    for temp in data:
        for i in temp:
            for j in temp:
                adjacency_matrix[cate2id[i], cate2id[j]] += 1

    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i] = adjacency_matrix[i] / adjacency_matrix[i, i]
        adjacency_matrix[i, i] = 0.0

    np.save(graph_path, adjacency_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, choices=['mscoco', 'vg500'])
    args = parser.parse_args()

    preprocessing_for_ssgrl(args.data)