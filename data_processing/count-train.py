import json
import os
from collections import Counter
import numpy as np


def get_labels(dir_dataset):
    labels = []

    batch_dirs = os.listdir(os.path.join(dir_dataset, 'train'))
    for batch_dir in batch_dirs:
        file_names = os.listdir(os.path.join(dir_dataset, 'train', batch_dir))
        for file_name in file_names:
            file_path = os.path.join(dir_dataset, 'train', batch_dir, file_name)
            data = np.load(file_path)
            labs = data['labels']

            labels.append(labs)

    labels = np.concatenate(labels, axis=0)
    labels_dict = Counter(labels)
    labels = list(labels_dict.keys())
    counts = list(labels_dict.values())

    sorted_labels_counts = sorted(zip(labels, counts))
    labels, counts = zip(*sorted_labels_counts)
    counts = np.array(counts)
    labels = np.array(labels)

    return labels, counts


if __name__ == '__main__':
    name_list = []
    count_list = []
    print("---------------- mnist ----------------")
    dir_dataset = "../datasets/mnist/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('mnist')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max ", np.max(counts))
    print("counts_min ", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum ", np.sum(counts))

    print("---------------- fashion-mnist ----------------")
    dir_dataset = "../datasets/fashion-mnist/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('fashion-mnist')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max", np.max(counts))
    print("counts_min", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum", np.sum(counts))

    print("---------------- cifar-10 ----------------")
    dir_dataset = "../datasets/cifar-10/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('cifar-10')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max", np.max(counts))
    print("counts_min", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum", np.sum(counts))

    print("---------------- spots-10 ----------------")
    dir_dataset = "../datasets/spots-10/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('spots-10')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max", np.max(counts))
    print("counts_min", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum", np.sum(counts))

    print("---------------- caltech-101 ----------------")
    dir_dataset = "../datasets/caltech-101/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('caltech-101')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max", np.max(counts))
    print("counts_min", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum", np.sum(counts))

    print("---------------- caltech-256 ----------------")
    dir_dataset = "../datasets/caltech-256/imbalance/"
    labels, counts = get_labels(dir_dataset)
    name_list.append('caltech-256')
    count_list.append(list(counts))
    print("counts    :", counts.shape, counts.dtype)
    print("labels    :", labels.shape, labels.dtype)
    print("labels    :", list(labels))
    print("counts    :", list(counts))
    print("counts_max", np.max(counts))
    print("counts_min", np.min(counts))
    print("max/min    ", counts.max() / counts.min())
    print("counts_sum", np.sum(counts))

    # 存储 count_train JSON 字典
    result_dict = {k: [int(x) for x in v] for k, v in zip(name_list, count_list)}

    # 保存为 JSON 文件
    with open("../datasets/count_train.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
