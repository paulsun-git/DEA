import os

import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

dir_dataset = "../datasets/caltech-256/"
dir_results = "../datasets/caltech-256/"

list_cls = []   # 类别名（相对根目录的路径）
list_cnt = []   # 对应目录下的文件个数

for root, dirs, files in os.walk(dir_dataset, topdown=False):
    # 从最深目录开始遍历，topdown=False 保证先访问子目录
    num_files = len([f for f in files if os.path.isfile(os.path.join(root, f))])
    if num_files == 0:
        # 当前目录无文件，跳过
        continue
    # 计算相对路径并标准化分隔符
    rel_path = os.path.relpath(root, dir_dataset)
    rel_path = rel_path.replace(os.sep, "\\")   # 统一用反斜杠
    list_cls.append(rel_path)
    list_cnt.append(num_files)

# 数据基础信息
print(len(list_cls))
print(len(list_cnt))
print(min(list_cnt), max(list_cnt))
print("list_cls =", list_cls)
print("list_cnt =", list_cnt)

# 标签重排序
paired = sorted(zip(list_cls, list_cnt), key=lambda x: x[1], reverse=True)
list_cls, list_cnt = map(list, zip(*paired))
print("排序后 list_cls =", list_cls)
print("排序后 list_cnt =", list_cnt)

# 类别名称-类别数量-采样率
num_classes = len(list_cls)
class_name = list_cls
class_cnt = list_cnt
class_rate = list(np.exp(-3 * np.sqrt(np.array([np.linspace(0, 2, num_classes).tolist()])))[0])
print("原始样本数量：", len(class_name))
print("原始样本数量：", len(class_cnt))
print("原始样本数量：", len(class_rate))


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


# 逐类别处理数据
for i, name in enumerate(class_name):
    print(f"当前处理的类[{i}]：", name)
    data, labels = [], []
    dir_images = os.path.join(dir_dataset, name)
    img_names = os.listdir(dir_images)
    for img_name in tqdm(img_names):
        img = cv2.imread(os.path.join(dir_images, img_name))  # BGR
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        data.append(img)

    data = np.stack(data)
    labels = np.array([i for _ in range(len(data))])
    print("data", data.shape, data.dtype)
    print("labels", labels.shape, labels.dtype)

    # 数据采样-分割
    (data_train, labels_train, data_test, labels_test, data_val, labels_val) \
        = split_val_and_test(data, labels, random_state=925)
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

    dir_save = os.path.join(dir_results, "imbalance", "train", "01")
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    np.savez_compressed(os.path.join(dir_save, f"class{i}-train.npz"), data=data_train, labels=labels_train)
    dir_save = os.path.join(dir_results, "imbalance", "test", "01")
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    np.savez_compressed(os.path.join(dir_save, f"class{i}-test.npz"), data=data_test, labels=labels_test)
    dir_save = os.path.join(dir_results, "imbalance", "val", "01")
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    np.savez_compressed(os.path.join(dir_save, f"class{i}-val.npz"), data=data_val, labels=labels_val)



