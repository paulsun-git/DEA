import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import timm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. 加载 MAE ----------
mae_model = timm.create_model("vit_base_patch16_224",
                              pretrained=False, img_size=224,
                              checkpoint_path='E:\\Download\\MAE\\model.safetensors')

mae_model = mae_model.to(device)
mae_model.eval()
for p in mae_model.parameters():
    p.requires_grad = False


# ---------- 2. 把 0-1 numpy -> [-1,1] tensor ----------
def numpy_to_tensor(x):
    """x: (N,3,224,224) 0.0-1.0 -> tensor [-1,1] on device"""
    x = torch.from_numpy(x).float() * 2.0 - 1.0        # 归一化到 [-1,1]
    # DINOv2 训练时用的是 ImageNet 均值方差，但官方 Demo 显示 *直接输入 [-1,1] 即可

    return x.to(device)


# ---------- 3. 提取函数 ----------
def extract_mae(imgs_np, labs_np, batch_size=200):
    """
    imgs_np: (N,3,224,224) float32 0.0~1.0
    labs_np: (N,)
    return: feats_np (N,768), labs_np
    """
    N = imgs_np.shape[0]
    feats = []

    for i in tqdm(range(0, N, batch_size), desc='MAE extract'):
        xb = numpy_to_tensor(imgs_np[i:i+batch_size]).to(device)
        if xb.size(2) != 224 or xb.size(3) != 224:
            xb = F.interpolate(xb, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # print('xb', xb.size())
            feat = mae_model.forward_features(xb)  # (B,197,768)
            feat = feat.mean(dim=1)  # (B,768)
            # print('feat', feat.size())

        feats.append(feat.cpu().numpy())

    feats_np = np.concatenate(feats, axis=0).astype(np.float32)

    return feats_np, labs_np


def get_feats_labs(dir_dataset, name_dataset):
    features, labels = [], []

    batch_dirs = os.listdir(os.path.join(dir_dataset, 'train'))
    for batch_dir in batch_dirs:
        file_names = os.listdir(os.path.join(dir_dataset, 'train', batch_dir))
        for file_name in file_names:
            file_path = os.path.join(dir_dataset, 'train', batch_dir, file_name)
            data = np.load(file_path)
            train_X = data['data']
            train_Y = data['labels']
            if name_dataset in ["spots-10"]:
                train_X = np.concatenate([train_X] * 3, axis=1)

            feats, labs = extract_mae(train_X, train_Y)

            features.append(feats)
            labels.append(labs)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


if __name__ == '__main__':
    print("---------------- cifar-10 ----------------")
    dir_dataset = "../datasets/cifar-10/imbalance/"
    features, labels = get_feats_labs(dir_dataset, name_dataset="cifar-10")
    print("features  :", features.shape, features.dtype)
    print("labels    :", labels.shape, labels.dtype)
    np.savez_compressed(os.path.join(dir_dataset, 'mae_features_labels-train.npz'), features=features, labels=labels)

    print("---------------- spots-10 ----------------")
    dir_dataset = "../datasets/spots-10/imbalance/"
    features, labels = get_feats_labs(dir_dataset, name_dataset="spots-10")
    print("features  :", features.shape, features.dtype)
    print("labels    :", labels.shape, labels.dtype)
    np.savez_compressed(os.path.join(dir_dataset, 'mae_features_labels-train.npz'), features=features, labels=labels)

    print("---------------- caltech-101 ----------------")
    dir_dataset = "../datasets/caltech-101/imbalance/"
    features, labels = get_feats_labs(dir_dataset, name_dataset="caltech-101")
    print("features  :", features.shape, features.dtype)
    print("labels    :", labels.shape, labels.dtype)
    np.savez_compressed(os.path.join(dir_dataset, 'mae_features_labels-train.npz'), features=features, labels=labels)

    print("---------------- caltech-256 ----------------")
    dir_dataset = "../datasets/caltech-256/imbalance/"
    features, labels = get_feats_labs(dir_dataset, name_dataset="caltech-256")
    print("features  :", features.shape, features.dtype)
    print("labels    :", labels.shape, labels.dtype)
    np.savez_compressed(os.path.join(dir_dataset, 'mae_features_labels-train.npz'), features=features, labels=labels)
