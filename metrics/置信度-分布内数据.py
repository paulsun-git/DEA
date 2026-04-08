import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def get_overall(label, pred, p, num_classes):
    """
    计算绘制分类结果所需数据

    :param label: 样本标签
    :param pred: 样本分类结果
    :param p: 样本分类置信度
    :return: sample_indices, is_correct, accuracy, correct_mean_un, incorrect_mean_un
    """
    # 检查数据长度是否一致
    if len(label) != len(pred) or len(label) != len(p):
        raise ValueError("The length of [label, pred, p] are not equal!")

    # 计算整体正确率
    acc_overall = metrics.accuracy_score(label, pred)

    # 创建字典存储结果
    acc_dict = {label_item: 0.0 for label_item in [lab for lab in range(num_classes)]}
    p_correct_dict = {label_item: None for label_item in [lab for lab in range(num_classes)]}
    p_incorrect_dict = {label_item: None for label_item in [lab for lab in range(num_classes)]}

    # 获取每个类别的信息
    for c in range(num_classes):
        c_indices = np.where(label == c)[0]
        c_label = label[c_indices]
        c_pred = pred[c_indices]
        c_p = p[c_indices]

        # 判断分类是否正确
        is_correct = c_label == c_pred

        # 计算分类正确率
        acc = np.mean(is_correct)

        # 存储结果
        acc_dict[c] = acc
        p_correct_dict[c] = c_p[is_correct]
        p_incorrect_dict[c] = c_p[~is_correct]

    return acc_dict, p_correct_dict, p_incorrect_dict, acc_overall


def plt_overall(acc_dict, p_correct_dict, p_incorrect_dict, acc_overall):
    """
    绘制分类结果 样本-正确率-不可信度 整体分布情况

    :return:
    """
    # ---------- 0. 准备数据 ----------
    classes = sorted(acc_dict.keys())  # 保证 0,1,2,...,n-1
    n_class = len(classes)

    # 每个类的样本量
    n_correct = np.array([len(p_correct_dict[c]) for c in classes])
    n_incorrect = np.array([len(p_incorrect_dict[c]) for c in classes])
    n_total = n_correct + n_incorrect

    # 横轴宽度按样本量比例分配
    width_ratio = n_total / n_total.sum()
    edges = np.concatenate([[0], np.cumsum(width_ratio)])  # 每段边界 0~1

    # ---------- 1. 建立画布 ----------
    fig = plt.figure(figsize=(10, 6))
    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    ax_prob = fig.add_subplot(gs[0])  # 上：概率散点
    ax_acc = fig.add_subplot(gs[1])  # 下：accuracy 柱

    # ---------- 2. 画概率散点 ----------
    correct_x, correct_y = [], []
    incorrect_x, incorrect_y = [], []
    for idx, c in enumerate(classes):
        left, right = edges[idx], edges[idx + 1]
        # 正确样本
        y_vals = p_correct_dict[c]
        x_vals = left + (right - left) * np.random.rand(len(y_vals))
        if len(y_vals) > 0:
            correct_x.append(np.mean(x_vals))
            correct_y.append(np.mean(y_vals))
        ax_prob.scatter(x_vals, y_vals, c='blue', s=15, alpha=0.6, label='correct' if idx == 0 else "")
        # 错误样本
        y_vals = p_incorrect_dict[c]
        x_vals = left + (right - left) * np.random.rand(len(y_vals))
        if len(y_vals) > 0:
            incorrect_x.append(np.mean(x_vals))
            incorrect_y.append(np.mean(y_vals))
        ax_prob.scatter(x_vals, y_vals, c='red', s=15, alpha=0.9, label='incorrect' if idx == 0 else "")

    # 拟合
    slope_correct, intercept_correct = np.polyfit(np.array(correct_x), np.array(correct_y), 1)
    Y_correct = slope_correct * np.array(correct_x) + intercept_correct
    slope_incorrect, intercept_incorrect = np.polyfit(np.array(incorrect_x), np.array(incorrect_y), 1)
    Y_incorrect = slope_incorrect * np.array(incorrect_x) + intercept_incorrect

    # ax_prob.plot(np.array(correct_x), np.array(Y_correct), c='blue', linestyle='--')
    # ax_prob.plot(np.array(incorrect_x), np.array(Y_incorrect), c='red', linestyle='--')

    # 分隔虚线
    for x in edges[1:-1]:
        ax_prob.axvline(x, color='gray', linestyle='--', linewidth=0.8)

    # 坐标轴装饰
    ax_prob.set_title(f'Accuracy, acc={acc_overall}, acc_={np.mean(list(acc_dict.values()))}')
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel('Uncertainty')
    ax_prob.legend(loc='upper right')
    # 把刻度改成类别编号
    ax_prob.set_xticks((edges[:-1] + edges[1:]) / 2)
    ax_prob.set_xticklabels(classes)

    # ---------- 3. 画 accuracy 柱状图 ----------
    ax_acc.scatter(classes, [acc_dict[c] for c in classes],
                   color='steelblue', marker='o', label='accuracy')
    y_min = min(list(acc_dict.values()))
    y_max = max(list(acc_dict.values()))
    y_min = max(y_min - 0.1, 0.0)
    y_max = min(y_max + 0.1, 1.0)
    ax_acc.set_ylim(y_min, y_max)
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Class')
    ax_acc.set_xticks(classes)

    # plt.savefig(f'分布内数据-acc-{dataset_name}-{method_name}.svg', format='svg')
    plt.show()


def get_brier(label_test, prob_test):
    N, C = prob_test.shape

    # 1. 整体 Brier 分数
    y_true_1hot = np.eye(C)[label_test]  # (N,C)
    overall = float(np.mean(np.sum((prob_test - y_true_1hot) ** 2, axis=1)))

    # 2. 每个类别的 Brier 分数
    per_class = []
    for c in range(C):
        # 该类样本掩码
        mask = (label_test == c)  # (N,)
        if mask.sum() == 0:  # 该类无样本
            per_class.append(np.nan)
            continue
        # 提取对应概率
        prob_c = prob_test[mask, c]  # (Nc,)
        # Brier 分数 = (1 - prob_c)^2 的均值
        brier_c = float(np.mean((1 - prob_c) ** 2))
        per_class.append(brier_c)

    # 3. 组装输出
    overall_brier = overall

    return per_class, overall_brier


def get_macro_auc(label_test, pred_test, prob_test, num_classes):
    per_class_prc1, per_class_prc2 = [], []

    # 计算每一个类别的 ROC
    label_ones = np.eye(num_classes)[label_test]
    for c in range(num_classes):
        # 真实标签位置的概率
        c_label = label_ones[:, c]
        c_score = prob_test[:, c]

        au_prc1 = metrics.average_precision_score(c_label, c_score)
        per_class_prc1.append(au_prc1)

        # 分类概率
        c_label = (label_test[label_test == c] == pred_test[label_test == c]).astype(int)
        c_score = np.max(prob_test, axis=1)[label_test == c]

        au_prc2 = metrics.average_precision_score(c_label, c_score)
        per_class_prc2.append(au_prc2)

    overall_prc1 = np.sum(per_class_prc1) / len(per_class_prc1)
    overall_prc2 = np.sum(per_class_prc2) / len(per_class_prc2)

    return per_class_prc1, per_class_prc2, overall_prc1, overall_prc2


def plt_confidence(per_class_prc1, overall_prc1, per_class_prc2, overall_prc2, per_class_brier, overall_brier):
    # 创建绘图
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5, 9), gridspec_kw={'height_ratios': [3, 3, 3]})

    # 绘制 macro-AUC
    axes[0].scatter([lab for lab in range(len(per_class_prc1))], per_class_prc1, color='blue', label='roc-class')

    y_min = min(per_class_prc1)
    y_max = max(per_class_prc1)
    y_min = max(y_min - 0.05, 0.0)
    y_max = min(y_max + 0.05, 1.0)
    if not np.isfinite(y_min):
        y_min = 0.0
    if not np.isfinite(y_max):
        y_max = 1.0
    axes[0].set_title(f'Macro-PRC1={overall_prc1:.5f}')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('ROC')
    axes[0].set_ylim(y_min, y_max)
    axes[0].legend()

    axes[1].scatter([lab for lab in range(len(per_class_prc2))], per_class_prc2, color='blue', label='prc-class')

    y_min = min(per_class_prc2)
    y_max = max(per_class_prc2)
    y_min = max(y_min - 0.05, 0.0)
    y_max = min(y_max + 0.05, 1.0)
    if not np.isfinite(y_min):
        y_min = 0.0
    if not np.isfinite(y_max):
        y_max = 1.0
    axes[1].set_title(f'Macro-PRC2={overall_prc2:.5f}')
    axes[1].set_xlabel('Class Label')
    axes[1].set_ylabel('PRC')
    axes[1].set_ylim(y_min, y_max)
    axes[1].legend()

    # 绘制 brier 分数
    axes[2].scatter([lab for lab in range(len(per_class_brier))], per_class_brier, color='blue', label='brier-class')

    y_min = min(per_class_brier)
    y_max = max(per_class_brier)
    y_min = max(y_min - 0.05, 0.0)
    y_max = min(y_max + 0.05, 1.0)
    axes[2].set_title(f'Brier={overall_brier:.5f}')
    axes[2].set_xlabel('Class Label')
    axes[2].set_ylabel('Brier')
    axes[2].set_ylim(y_min, y_max)
    axes[2].legend()

    # 显示图形
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'分布内数据-auc-{dataset_name}-{method_name}.svg', format='svg')

    plt.show()


global dataset_name
global method_name

if __name__ == '__main__':
    file_path_test = f"../results/test/"

    # 读取测试集推理结果
    label_test = np.load(os.path.join(file_path_test, 'label_test.npy'))
    pred_test = np.load(os.path.join(file_path_test, 'pred_test.npy'))
    prob_test = np.load(os.path.join(file_path_test, 'prob_test.npy'))
    un_test = np.load(os.path.join(file_path_test, 'uncertainty_test.npy'))
    p = np.max(prob_test, axis=1)
    num_classes = prob_test.shape[1]

    # 数据处理
    label_test = np.squeeze(label_test)
    pred_test = np.squeeze(pred_test)

    # 计算整体分布所需数据
    acc_dict, un_correct_dict, un_incorrect_dict, acc_overall = get_overall(label_test, pred_test, un_test, num_classes)

    # 绘制结果图 1
    plt_overall(acc_dict, un_correct_dict, un_incorrect_dict, acc_overall)
    print("Acc of each class:", list(acc_dict.values()))

    # Brier 分数
    per_class_brier, overall_brier = get_brier(label_test, prob_test)
    print("Brier of each class:", list(per_class_brier))

    # 宏平均 auc
    per_class_prc1, per_class_prc2, overall_prc1, overall_prc2 = get_macro_auc(label_test, pred_test, prob_test, num_classes)
    print("PRC1 of each class:", list(per_class_prc1))
    print("PRC2 of each class:", list(per_class_prc2))

    # 绘制结果图 2
    plt_confidence(per_class_prc1, overall_prc1,
                   per_class_prc2, overall_prc2,
                   per_class_brier, overall_brier)

