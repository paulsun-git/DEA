import os

import torchvision.transforms as transforms
from collections import Counter
import numpy as np
import torchvision

dir_dataset = "../datasets/mnist/"

# 加载数据集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root=dir_dataset, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root=dir_dataset, train=False, download=True, transform=transform)

len_train_dataset = len(train_dataset)
len_test_dataset = len(test_dataset)

# 提取数据，转为 numpy 数据
data_train, labels_train = [], []
for i in range(len_train_dataset):
    img, tgt = train_dataset[i]
    img = img.unsqueeze(0)
    img = img.squeeze().numpy()
    data_train.append(img)
    labels_train.append(tgt)

data_test, labels_test = [], []
for i in range(len_test_dataset):
    img, tgt = test_dataset[i]
    img = img.unsqueeze(0)
    img = img.squeeze().numpy()
    data_test.append(img)
    labels_test.append(tgt)

data_train = np.array(data_train, dtype=np.float32)
labels_train = np.array(labels_train, dtype=np.int32)
data_test = np.array(data_test, dtype=np.float32)
labels_test = np.array(labels_test, dtype=np.int32)

data_train = np.expand_dims(data_train, axis=1)
data_test = np.expand_dims(data_test, axis=1)

labels_counts_dict_train = dict(Counter(labels_train))
labels_counts_dict_test = dict(Counter(labels_test))

print("数据基础信息：")
print("data_train", data_train.shape, data_train.dtype)
print("labels_train", labels_train.shape, labels_train.dtype)
print("data_test", data_test.shape, data_test.dtype)
print("labels_test", labels_test.shape, labels_test.dtype)
print("-----------------------------------------------------------------")
print("数据详细信息：")
print("range of data_train", np.min(data_train), np.max(data_train))
print("range of data_test", np.min(data_test), np.max(data_test))
print("count of labels_train", labels_counts_dict_train)
print("count of labels_test", labels_counts_dict_test)
print("-----------------------------------------------------------------")


# 标签转换，按照训练数据样本数从大到小排序，新标签编号 0-class_num
def reorder_labels(labels_train, labels_test):
    """
    根据训练集中各类别样本数量重新排序标签

    参数:
    labels_train: 训练集标签
    labels_test: 测试集标签

    返回:
    labels_train_new: 重新排序后的训练集标签
    labels_test_new: 重新排序后的测试集标签
    label_mapping: 新旧标签映射字典
    """
    # 计算训练集中各类别数量
    unique_labels, counts = np.unique(labels_train, return_counts=True)
    labels_counts = dict(zip(unique_labels, counts))

    # 按数量排序（降序）
    sorted_items = sorted(labels_counts.items(), key=lambda x: x[1], reverse=True)

    # 创建映射关系
    old_to_new = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_items)}

    # 转换标签
    labels_train_new = np.array([old_to_new[label] for label in labels_train])
    labels_test_new = np.array([old_to_new[label] for label in labels_test])

    return labels_train_new, labels_test_new, old_to_new


labels_train_new, labels_test_new, old_to_new = reorder_labels(labels_train, labels_test)
labels_counts_dict_train_new = dict(Counter(labels_train_new))
labels_counts_dict_test_new = dict(Counter(labels_test_new))
print("标签转换结果：")
print("old labels_train", labels_counts_dict_train)
print("new labels_train", labels_counts_dict_train_new)
print("old labels_test", labels_counts_dict_test)
print("new labels_test", labels_counts_dict_test_new)
print(labels_test[:20])
print(labels_test_new[:20])
print(labels_train[:20])
print(labels_train_new[:20])
print("-----------------------------------------------------------------")
labels_train = labels_train_new
labels_test = labels_test_new


# 采样获取验证集
def split_val_like_test(data_train, labels_train, data_test, labels_test, random_state=925):
    """
    从训练集中不放回地抽取与测试集类别分布一致的验证集。

    参数
    ----
    data_train : np.ndarray
    labels_train : np.ndarray
    data_test  : np.ndarray
    labels_test : np.ndarray
    random_state : int, 随机种子

    返回
    ----
    data_train, labels_train,   # 剩余训练集
    data_test,  labels_test,    # 原测试集
    data_val,   labels_val      # 新验证集
    """
    rng = np.random.default_rng(random_state)

    # 计算测试集每类样本数
    uniq_labels, test_counts = np.unique(labels_test, return_counts=True)
    needed = dict(zip(uniq_labels, test_counts))

    # 按类别分组训练集索引
    train_idx_by_label = {l: np.where(labels_train == l)[0] for l in uniq_labels}

    # 检查训练集每类数量是否足够
    for l, need in needed.items():
        have = len(train_idx_by_label[l])
        if have < need:
            raise ValueError(
                f'类别 {l} 训练集样本不足：需要 {need}，仅有 {have}')

    # 不放回随机抽取
    val_idx = []
    for l, need in needed.items():
        picked = rng.choice(train_idx_by_label[l], size=need, replace=False)
        val_idx.append(picked)
    val_idx = np.concatenate(val_idx)

    # 剩余训练集索引
    train_mask = np.ones(len(labels_train), dtype=bool)
    train_mask[val_idx] = False
    remain_idx = np.where(train_mask)[0]

    # 构建新数据集
    data_val = data_train[val_idx]
    labels_val = labels_train[val_idx]

    data_train = data_train[remain_idx]
    labels_train = labels_train[remain_idx]

    return data_train, labels_train, data_test, labels_test, data_val, labels_val


(data_train, labels_train, data_test, labels_test, data_val, labels_val) \
    = split_val_like_test(data_train, labels_train, data_test, labels_test, random_state=925)
print("standard 训练集-测试集-验证集：")
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

dir_save = os.path.join(dir_dataset, "standard", "train", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "mnist-train.npz"), data=data_train, labels=labels_train)
dir_save = os.path.join(dir_dataset, "standard", "test", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "mnist-test.npz"), data=data_test, labels=labels_test)
dir_save = os.path.join(dir_dataset, "standard", "val", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "mnist-val.npz"), data=data_val, labels=labels_val)
# 设置类别采样率，构建不平衡训练集
num_classes = len(np.unique(labels_train))
rate = list(np.exp(-3 * np.sqrt(np.array([np.linspace(0, 2, num_classes).tolist()])))[0])


def sample_by_rate(data_train, labels_train, rate, random_state=925):
    """
    按类别对应的采样率对训练集进行不放回随机采样。

    参数
    ----
    data_train  : np.ndarray
    labels_train: np.ndarray, 元素为 0..n-1
    rate        : list or np.ndarray, 长度 n, 每类采样率 ∈[0,1]
    random_state: int, 随机种子

    返回
    ----
    data_train_new, labels_train_new
    """
    rate = np.asarray(rate)
    if not np.all((rate >= 0) & (rate <= 1)):
        raise ValueError("rate 中的元素必须在 [0, 1] 区间内")

    rng = np.random.default_rng(random_state)
    new_data, new_labels = [], []

    for cls_id, r in enumerate(rate):
        # 当前类别所有索引
        cls_idx = np.where(labels_train == cls_id)[0]
        n_cls = len(cls_idx)
        if n_cls == 0:
            raise ValueError(f"训练集中不存在类别 {cls_id}")          # 训练集里没有这一类，报错
        # 计算要采的样本数，至少为 1 如果 rate>0
        n_pick = max(1, int(np.round(r * n_cls))) if r > 0 else 0
        n_pick = min(n_pick, n_cls)   # 防止舍入后越界
        if n_pick == 0:
            continue
        picked = rng.choice(cls_idx, size=n_pick, replace=False)
        new_data.append(data_train[picked])
        new_labels.append(labels_train[picked])

    return np.concatenate(new_data, axis=0), np.concatenate(new_labels, axis=0)


data_train, labels_train = sample_by_rate(data_train, labels_train, rate, random_state=925)
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
np.savez_compressed(os.path.join(dir_save, "mnist-train.npz"), data=data_train, labels=labels_train)
dir_save = os.path.join(dir_dataset, "imbalance", "test", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "mnist-test.npz"), data=data_test, labels=labels_test)
dir_save = os.path.join(dir_dataset, "imbalance", "val", "01")
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
np.savez_compressed(os.path.join(dir_save, "mnist-val.npz"), data=data_val, labels=labels_val)

