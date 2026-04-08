import json
import os
import numpy as np


def compute_intra_trace(feats, labs):
    """
    计算每个类的类内协方差矩阵的迹（trace）

    参数
    ----
    feats : ndarray, shape (N, d)
        样本特征矩阵
    labs : ndarray, shape (N,)
        样本标签（0 ~ K-1）

    返回
    ----
    intra_trace : ndarray, shape (K,)
        每个类的类内协方差矩阵的迹
    """
    labs = np.asarray(labs, dtype=int)
    K = labs.max() + 1
    intra_trace = np.zeros(K, dtype=feats.dtype)

    for k in range(K):
        idx = labs == k
        if idx.sum() < 2:           # 样本数不足，无法计算协方差
            intra_trace[k] = 0.0
            continue
        X_k = feats[idx]
        cov_k = np.cov(X_k, rowvar=False, bias=False)
        intra_trace[k] = np.trace(cov_k)

    return intra_trace


def compute_inter_centroid(feats, labs):
    """
    计算以类中心（原型向量）为基础的类间距离矩阵

    参数
    ----
    feats : ndarray, shape (N, d)
        样本特征矩阵
    labs : ndarray, shape (N,)
        样本标签（0 ~ K-1）

    返回
    ----
    inter_centroid : ndarray, shape (K, K)
        对称矩阵，inter_centroid[i, j] 为类 i 与类 j 中心之间的欧氏距离
    """
    labs = np.asarray(labs, dtype=int)
    K = labs.max() + 1
    centroids = np.zeros((K, feats.shape[1]), dtype=feats.dtype)
    inter_centroid = np.zeros((K, K), dtype=feats.dtype)

    # 计算各类中心
    for k in range(K):
        centroids[k] = feats[labs == k].mean(axis=0)

    for i in range(K):
        for j in range(i, K):
            # 计算类中心之间的欧氏距离
            d = np.linalg.norm(centroids[i] - centroids[j])

            # 计算类中心之间的余弦距离
            # dot = np.dot(centroids[i], centroids[j])
            # norm_i = np.linalg.norm(centroids[i])
            # norm_j = np.linalg.norm(centroids[j])
            # if norm_i == 0 or norm_j == 0:
            #     d = 0.0 if i == j else 1.0
            # else:
            #     cosine_sim = dot / (norm_i * norm_j)
            #     cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            #     d = 1.0 - cosine_sim

            # 计算类中心之间的曼哈顿距离
            # d = np.sum(np.abs(centroids[i] - centroids[j]))

            inter_centroid[i, j] = inter_centroid[j, i] = d

    return inter_centroid


def compute_difficulty(intra_trace, inter_centroid):
    """
    综合类内迹与类中心距离，计算每个类的分类难度

    难度指标：
        difficulty[k] = intra_trace[k] / min_{j≠k} inter_centroid[k, j]

    参数
    ----
    feats, labs : 同上

    返回
    ----
    difficulty : ndarray, shape (K,)
        每个类的分类难度（越大越难分）
    """
    K = intra_trace.shape[0]
    difficulty = np.zeros(K, dtype=feats.dtype)

    for k in range(K):
        # 去掉对角线后取最小类中心距离
        inter_k = inter_centroid[k]
        inter_k = np.delete(inter_k, k)
        min_inter = inter_k.min() if len(inter_k) else 1.0  # 避免除零
        difficulty[k] = (intra_trace[k] / min_inter)

    return difficulty


def get_dc(D_inter, D_intra, num_classes):
    Dc = []
    for cls in range(num_classes):
        row_vector = D_inter[cls]

        diagonal_value = D_intra[cls]
        modified_vector = np.delete(row_vector, cls)

        if diagonal_value == 0:
            raise ValueError(f"Diagonal value is zero, cannot perform division.")

        divided_vector = modified_vector / diagonal_value
        vector_mod = np.linalg.norm(divided_vector)

        Dc.append(vector_mod)

    Dc = np.array(Dc)

    return Dc


if __name__ == '__main__':
    name_list = []
    dc_list = []
    print("---------------- mnist ----------------")
    dir_dataset = "../datasets/mnist/imbalance/"
    file_path = os.path.join(dir_dataset, "train", "01", "mnist-train.npz")
    data = np.load(file_path)
    feats = data['data'].reshape(data['data'].shape[0], -1)  # mnist 和 fashion-mnist 特征简单直接计算即可
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 10
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('mnist')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    print("---------------- fashion-mnist ----------------")
    dir_dataset = "../datasets/fashion-mnist/imbalance/"
    file_path = os.path.join(dir_dataset, "train", "01", "fashion_mnist-train.npz")
    data = np.load(file_path)
    feats = data['data'].reshape(data['data'].shape[0], -1)  # mnist 和 fashion-mnist 特征简单直接计算即可
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 10
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('fashion_mnist')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    print("---------------- cifar-10 ----------------")
    dir_dataset = "../datasets/cifar-10/imbalance/"
    file_path = os.path.join(dir_dataset, "dino_features_labels-train.npz")
    data = np.load(file_path)
    feats = data['features']
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 10
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('cifar-10')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    print("---------------- spots-10 ----------------")
    dir_dataset = "../datasets/spots-10/imbalance/"
    file_path = os.path.join(dir_dataset, "dino_features_labels-train.npz")
    data = np.load(file_path)
    feats = data['features']
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 10
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('spots-10')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    print("---------------- caltech-101 ----------------")
    dir_dataset = "../datasets/caltech-101/imbalance/"
    file_path = os.path.join(dir_dataset, "dino_features_labels-train.npz")
    data = np.load(file_path)
    feats = data['features']
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 101
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('caltech-101')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    print("---------------- caltech-256 ----------------")
    dir_dataset = "../datasets/caltech-256/imbalance/"
    file_path = os.path.join(dir_dataset, "dino_features_labels-train.npz")
    data = np.load(file_path)
    feats = data['features']
    labs = data['labels']
    print("feats    :", feats.shape, feats.dtype)
    print("labs     :", labs.shape, labs.dtype)

    num_classes = 256
    intra_trace = compute_intra_trace(feats, labs)
    inter_centroid = compute_inter_centroid(feats, labs)

    Dc = compute_difficulty(intra_trace, inter_centroid)
    name_list.append('caltech-256')
    dc_list.append(list(Dc))
    print("Dc : ", list(Dc))

    # 存储 Dc JSON 字典
    result_dict = {k: [float(x) for x in v] for k, v in zip(name_list, dc_list)}

    # 保存为 JSON 文件
    with open("../datasets/Dc_dino.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
