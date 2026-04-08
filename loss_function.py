import json

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
    intra_trace = np.zeros(K, dtype=np.float32)

    for k in range(K):
        idx = labs == k
        if idx.sum() < 2:
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
    inter_centroid = np.zeros((K, K), dtype=np.float32)

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
    difficulty = np.zeros(K, dtype=np.float32)

    for k in range(K):
        # 去掉对角线后取最小类中心距离
        inter_k = inter_centroid[k]
        inter_k = np.delete(inter_k, k)
        min_inter = inter_k.min() if len(inter_k) else 1.0
        difficulty[k] = (intra_trace[k] / min_inter)

    return difficulty


def get_dc(data_name):
    # 从JSON文件加载所有dc值
    with open("./datasets/Dc_dino.json", "r", encoding="utf-8") as f:
        dc_data = json.load(f)

    # 检查数据集名称是否有效
    if data_name not in dc_data:
        raise ValueError("Error Dataset Name!")

    # 转换为torch tensor返回
    dc = torch.tensor(dc_data[data_name])

    return dc


def get_eta(data_name):
    # 从JSON文件加载所有eta值
    with open("./datasets/count_train.json", "r", encoding="utf-8") as f:
        eta_data = json.load(f)

    # 检查数据集名称是否有效
    if data_name not in eta_data:
        raise ValueError("Error Dataset Name!")

    # 转换为torch tensor返回
    eta = torch.tensor(eta_data[data_name])
    eta = torch.sum(eta) / eta

    return eta


def get_device():
    """
    Get the Current Device
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_kl_loss(alphas, labels, target_concentration=1.0, concentration=1.0, epsilon=1e-8):
    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    target_alphas += torch.zeros_like(alphas).scatter_(-1, (labels.unsqueeze(-1)).long(), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss_kl = torch.squeeze(alp0_term + alphas_term)

    return loss_kl


def get_loss(dcc, alpha, w, xi, label, epoch_num, annealing_step, mu, sigma, dataset_name, device=None):
    """
    Loss of Model
    """
    if not device:
        device = get_device()

    if dcc is not None:
        dcc = dcc.to(device)
    alpha = alpha.to(device)
    xi = xi.to(device)
    label = label.to(device)

    # 求 edl-loss
    S = torch.sum(alpha, dim=-1, keepdim=True)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - label) + 1
    kl_div = annealing_coef * compute_kl_loss(kl_alpha, torch.argmax(label, dim=1))

    all_loss = A + kl_div

    # 类损失加权
    labels = torch.argmax(label, dim=1)
    labels = labels.to(torch.int32)
    unique_labels = torch.unique(labels)

    dc = get_dc(dataset_name).to(device)

    sigma = torch.tensor(sigma, dtype=torch.float32)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    t = (torch.exp(annealing_coef * sigma) - 1) / (torch.exp(sigma) - 1)

    if dcc is not None:
        Dc = (dc * (1 - t) + dcc * t)
    else:
        Dc = (dc * (1 - t))

    Dc = Dc / Dc.sum()

    loss = 0.0
    for lab in unique_labels:
        index = labels == lab
        item_loss = all_loss[index]
        it = torch.mean(item_loss)

        loss += (Dc[lab]) * it

    loss = loss / len(unique_labels)

    # 记忆校准模块约束项
    w_sum = torch.sum(w * label, dim=0)
    denom = label.sum(0).long()
    w_mean = w_sum[denom != 0] / denom[denom != 0]

    kl_w = (w_mean / torch.sum(w_mean)) + 1e-8
    kl_xi = (xi[denom != 0] / torch.sum(xi[denom != 0])) + 1e-8
    kl_ecm = F.kl_div(kl_w.log(), kl_xi, reduction='sum')

    # 总损失
    loss += mu * kl_ecm

    return loss
