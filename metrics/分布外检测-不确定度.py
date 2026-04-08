import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def get_overall_un(label_test, pred_test, un_test, un_ood):
    """
    计算绘制分类结果所需数据

    """
    # 创建样本索引
    sample_indices = np.arange(1, len(label_test) + 1)

    # 判断分类是否正确
    is_correct = label_test == pred_test

    # 计算分类正确和分类错误样本的不可信度平均值
    un_correct = un_test[is_correct]
    un_incorrect = un_test[~is_correct]

    label = np.concatenate((np.array([0] * un_test.shape[0]), np.array([1] * un_ood.shape[0])))
    score = np.concatenate((un_test, un_ood))

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    au_roc = metrics.auc(fpr, tpr)
    au_prc = metrics.average_precision_score(label, score)

    return un_correct, un_incorrect, un_ood, fpr, tpr, au_roc, au_prc


def get_overall_maxp(label_test, pred_test, prob_test, prob_ood):
    """
    计算绘制分类结果所需数据

    """
    # 创建样本索引
    sample_indices = np.arange(1, len(label_test) + 1)

    # 判断分类是否正确
    is_correct = label_test == pred_test

    # 获得 max(p)
    prob_test = np.max(prob_test, axis=1)
    prob_ood = np.max(prob_ood, axis=1)

    # 计算分类正确和分类错误样本的不可信度平均值
    prob_correct = prob_test[is_correct]
    prob_incorrect = prob_test[~is_correct]

    label = np.concatenate((np.array([1] * prob_test.shape[0]), np.array([0] * prob_ood.shape[0])))
    score = np.concatenate((prob_test, prob_ood))

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    au_roc = metrics.auc(fpr, tpr)
    au_prc = metrics.average_precision_score(label, score)

    return prob_correct, prob_incorrect, prob_ood, fpr, tpr, au_roc, au_prc


def plt_overall(un_correct, un_incorrect, un_ood, fpr, tpr, au_roc, au_prc):
    """
    绘制分类结果 样本-正确率-不可信度 整体分布情况

    """
    # 创建绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 9), gridspec_kw={'height_ratios': [4, 5]})

    # 绘制 uncertainty 分布
    x1 = np.linspace(1, 10, len(un_ood))
    x2 = np.linspace(1, 10, len(un_correct))
    x3 = np.linspace(1, 10, len(un_incorrect))

    axes[0].scatter(x1, un_ood, color='#0081cf', alpha=0.5, s=3)
    axes[0].scatter(x2, un_correct, color='#30eb21', alpha=0.7, s=3)
    axes[0].scatter(x3, un_incorrect, color='#eb214a', alpha=0.9, s=3)
    axes[0].plot([0, 10], [np.mean(un_ood)] * 2, color='#0081cf')
    axes[0].plot([0, 10], [np.mean(un_correct)] * 2, color='#30eb21')
    axes[0].plot([0, 10], [np.mean(un_incorrect)] * 2, color='#eb214a')

    axes[0].set_title(f'Uncertainty of Test and OOD')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Uncertainty')
    axes[0].legend()

    # 绘制 AUC
    axes[1].plot(fpr, tpr, color='red', label='micro-roc')

    axes[1].set_title(f'Micro-ROC={au_roc:.5f}, Micro-PRC={au_prc:.5f}')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()

    # 显示图形
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"分布外数据-{dataset_name}-{method_name}.svg", format='svg')

    plt.show()


if __name__ == '__main__':
    file_path_test = f'../results/test/'
    file_path_ood = f'../results/ood/'

    # 读取测试集推理结果
    label_test = np.load(os.path.join(file_path_test, 'label_test.npy'))
    pred_test = np.load(os.path.join(file_path_test, 'pred_test.npy'))
    prob_test = np.load(os.path.join(file_path_test, 'prob_test.npy'))
    un_test = np.load(os.path.join(file_path_test, 'uncertainty_test.npy'))

    prob_ood = np.load(os.path.join(file_path_ood, 'prob_test.npy'))
    un_ood = np.load(os.path.join(file_path_ood, 'uncertainty_test.npy'))

    # 数据处理
    label_test = np.squeeze(label_test)
    pred_test = np.squeeze(pred_test)

    # 计算整体分布所需数据
    un_correct, un_incorrect, un_ood, fpr, tpr, au_roc, au_prc = get_overall_un(label_test, pred_test, un_test, un_ood)

    # 绘制结果图
    plt_overall(un_correct, un_incorrect, un_ood, fpr, tpr, au_roc, au_prc)



