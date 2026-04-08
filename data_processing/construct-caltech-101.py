import os

import torchvision.transforms as transforms
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision


dir_dataset = "../datasets/caltech-101/"

# 加载数据集
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.Grayscale(3),
                                transforms.ToTensor()])
all_dataset = torchvision.datasets.Caltech101(root=dir_dataset, download=False, transform=transform)

len_all_dataset = len(all_dataset)

# 提取数据，转为 numpy 数据
data_all, labels_all = [], []
for i in range(len_all_dataset):
    img, tgt = all_dataset[i]
    img = img.unsqueeze(0)
    img = img.squeeze().numpy()
    data_all.append(img)
    labels_all.append(tgt)

data_all = np.array(data_all, dtype=np.float32)
labels_all = np.array(labels_all, dtype=np.int32)

# data_train = np.expand_dims(data_train, axis=1)
# data_test = np.expand_dims(data_test, axis=1)

labels_counts_dict_all = dict(Counter(labels_all))

print("数据基础信息：")
print("data_all", data_all.shape, data_all.dtype)
print("labels_all", labels_all.shape, labels_all.dtype)
print("-----------------------------------------------------------------")
print("数据详细信息：")
print("range of data_all", np.min(data_all), np.max(data_all))
print("count of labels_all", labels_counts_dict_all)
print("-----------------------------------------------------------------")


# 标签转换，按照训练数据样本数从大到小排序，新标签编号 0-class_num
def reorder_labels(labels_all):
    """
    根据数据集中各类别样本数量重新排序标签

    参数:
    labels_all: 数据集标签

    返回:
    labels_all_new: 重新排序后的数据集标签
    label_mapping: 新旧标签映射字典
    """
    # 计算数据集中各类别数量
    unique_labels, counts = np.unique(labels_all, return_counts=True)
    labels_counts = dict(zip(unique_labels, counts))

    # 按数量排序（降序）
    sorted_items = sorted(labels_counts.items(), key=lambda x: x[1], reverse=True)

    # 创建映射关系
    old_to_new = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_items)}

    # 转换标签
    labels_all_new = np.array([old_to_new[label] for label in labels_all])

    return labels_all_new, old_to_new


labels_all_new, old_to_new = reorder_labels(labels_all)
labels_counts_dict_all_new = dict(Counter(labels_all_new))
print("标签转换结果：")
print("old labels_all", labels_counts_dict_all)
print("new labels_all", labels_counts_dict_all_new)
print(labels_all[:20])
print(labels_all_new[:20])
print("-----------------------------------------------------------------")
labels_all = labels_all_new


# 采样获取验证集、测试集
def split_val_and_test(data_all, labels_all, random_state=925):
    """
    从数据集中不放回地抽取验证集和测试集

    参数
    ----
    data_all : np.ndarray
    labels_all : np.ndarray
    random_state : int, 随机种子

    返回
    ----
    data_train, labels_train,   # 训练集
    data_test,  labels_test,    # 测试集
    data_val,   labels_val      # 验证集
    """
    # 第一次拆分：先分出 80% 训练+验证，20% 测试
    data_train_val, data_test, labels_train_val, labels_test = train_test_split(
        data_all, labels_all,
        test_size=0.2,
        random_state=random_state,
        stratify=labels_all
    )

    # 第二次拆分：把 80% 再拆成 60% 训练 + 20% 验证
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_train_val, labels_train_val,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=random_state,
        stratify=labels_train_val
    )

    return data_train, labels_train, data_test, labels_test, data_val, labels_val


(data_train, labels_train, data_test, labels_test, data_val, labels_val) \
    = split_val_and_test(data_all, labels_all, random_state=925)
print("imbalance 训练集-测试集-验证集：")
print("data_train", data_train.shape, data_train.dtype)
print("labels_train", labels_train.shape, labels_train.dtype)
print("data_test", data_test.shape, data_test.dtype)
print("labels_test", labels_test.shape, labels_test.dtype)
print("data_val", data_val.shape, data_val.dtype)
print("labels_val", labels_val.shape, labels_val.dtype)
print("-----------------------------------------------------------------")
print("count of labels_train", dict(Counter(labels_train)))
print("count of labels_test", dict(Counter(labels_test)))
print("count of labels_val", dict(Counter(labels_val)))
print("-----------------------------------------------------------------")

dir_save = os.path.join(dir_dataset, "imbalance", "train", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "caltech_101-train.npz"), data=data_train, labels=labels_train)
dir_save = os.path.join(dir_dataset, "imbalance", "test", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "caltech_101-test.npz"), data=data_test, labels=labels_test)
dir_save = os.path.join(dir_dataset, "imbalance", "val", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "caltech_101-val.npz"), data=data_val, labels=labels_val)

