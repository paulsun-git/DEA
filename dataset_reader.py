import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CreateDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        自定义 Dataset 类
        :param data: NumPy 数组，形状为 (num_samples, channels, 224, 224)
        :param labels: NumPy 数组，形状为 (num_samples,)，表示每个样本的标签
        :param transform: 可选的转换函数，用于对数据进行预处理
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        else:
            return sample


def read_data(dir_data, file_names):
    data_X, data_Y = [], []
    class_num = 0
    for file_name in file_names:
        class_num += 1
        data = np.load(os.path.join(dir_data, file_name))
        X = data['data']
        Y = data['labels']
        
        data_X.append(X)
        data_Y.append(Y)

    data_X = np.concatenate(data_X, axis=0)
    data_Y = np.concatenate(data_Y, axis=0)

    return data_X, data_Y


def load_dataset_train(dir_data, data_name, i, batch_size=200):
    """
        :读取数据集
        :返回训练数据
    """
    file_names = os.listdir(os.path.join(dir_data, "train", f"0{i+1}"))
    train_X, train_Y = read_data(os.path.join(dir_data, "train", f"0{i+1}"), file_names)

    # 数据通道扩展
    if data_name in ["spots-10"]:
        train_X = np.concatenate([train_X] * 3, axis=1)

    # 设置图像大小转换
    if data_name in ["cifar-10", "cifar-100", "spots-10"]:
        transform = transforms.Compose([transforms.Lambda(lambda x: torch.from_numpy(x)), transforms.Resize((224, 224)), ])
    else:
        transform = None

    # 创建数据集
    train_dataset = CreateDataset(train_X, train_Y, transform=transform)
    g = torch.Generator()
    g.manual_seed(925)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    
    return train_dataset


def load_dataset_test(dir_data, data_name, i, batch_size=200):
    """
        :读取数据集
        :返回测试数据
    """
    file_names = os.listdir(os.path.join(dir_data, "test", f"0{i+1}"))
    test_X, test_Y = read_data(os.path.join(dir_data, "test", f"0{i+1}"), file_names)

    # 数据通道扩展
    if data_name in ["spots-10"]:
        test_X = np.concatenate([test_X] * 3, axis=1)

    # 设置图像大小转换
    if data_name in ["cifar-10", "cifar-100", "spots-10"]:
        transform = transforms.Compose(
            [transforms.Lambda(lambda x: torch.from_numpy(x)), transforms.Resize((224, 224)), ])
    else:
        transform = None

    # 创建数据集
    test_dataset = CreateDataset(test_X, test_Y, transform=transform)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataset


def load_dataset_val(dir_data, data_name, i, batch_size=200):
    """
        :读取数据集
        :返回验证数据
    """
    file_names = os.listdir(os.path.join(dir_data, "val", f"0{i+1}"))
    val_X, val_Y = read_data(os.path.join(dir_data, "val", f"0{i+1}"), file_names)

    # 数据通道扩展
    if data_name in ["spots-10"]:
        val_X = np.concatenate([val_X] * 3, axis=1)

    # 设置图像大小转换
    if data_name in ["cifar-10", "cifar-100", "spots-10"]:
        transform = transforms.Compose(
            [transforms.Lambda(lambda x: torch.from_numpy(x)), transforms.Resize((224, 224)), ])
    else:
        transform = None

    # 创建数据集
    val_dataset = CreateDataset(val_X, val_Y, transform=transform)
    val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_dataset

