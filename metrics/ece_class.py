import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def get_ECE(prob, pred, label, nBin):
    """
    计算 ECE 所需数据

    :param prob: 测试数据置信度列表
    :param pred: 测试数据预测结果
    :param label: 测试数据实际标签
    :param nBin: 置信度分区数量
    :return: bars, num_bars, acc_bars, conf_bars, ECE
    """
    # Confidence 分区间
    bars = np.linspace(0, 1, nBin + 1)

    acc_bars = []
    num_bars = []
    conf_bars = []
    ECE = 0.0

    # 求 acc 和 conf
    for i in range(1, len(bars)):
        y_label = []
        y_pred = []
        y_prob = []
        for j in range(len(label)):
            conf = max(prob[j, :])
            if bars[i - 1] < conf <= bars[i]:
                y_label.append(label[j])
                y_pred.append(pred[j])
                y_prob.append(conf)

        if len(y_label) == 0:
            acc_bars.append(0.0)
            conf_bars.append(0.0)
            num_bars.append(0)
        else:
            acc_bars.append(metrics.accuracy_score(y_label, y_pred))
            conf_bars.append(sum(y_prob) / len(y_prob))
            num_bars.append(len(y_label))

    # 求 ECE
    for i in range(nBin):
        ECE += (num_bars[i] / len(label)) * (np.abs(acc_bars[i] - conf_bars[i]))

    return bars, num_bars, acc_bars, conf_bars, ECE


def plt_ECE(bars, num_bars, acc_bars, num_samples, ECE):
    """
    # 绘制 ECE 结果图

    :param bars: 置信度分区区间
    :param num_bars: 置信度分区数量
    :param acc_bars: 分区内正确率列表
    :param conf_bars: 分区内置信度列表
    :param ECE: ECE 值
    :param dataset_name: 数据集名称
    :param save: 是否保存结果图像
    :param c: 置否存在类别选择
    :return:
    """
    # 计算每个区间的宽度
    bar_widths = np.diff(bars)  # 计算相邻值之间的差值

    # 计算每个条形的中心位置
    bar_centers = bars[:-1] + bar_widths / 2

    # 创建绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})

    axes[0].bar(bar_centers, bar_centers, width=bar_widths, color='#00d2fc', label='acc-expectation',
                align='center', edgecolor='black', alpha=0.8)
    axes[0].bar(bar_centers, acc_bars, width=bar_widths, color='#ff9671', label='acc-prediction',
                align='center', edgecolor='black', alpha=0.7)
    axes[0].plot(bar_centers, bar_centers, color='red', linestyle='--')

    axes[0].set_title(f'Expected Calibration Error, ece = {ECE:.5f}')
    axes[0].set_xticks(bars)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].bar(bar_centers, np.array(num_bars) / num_samples, width=bar_widths, color='#2c73d2', label='confidence',
                align='center', edgecolor='black', alpha=0.9)

    axes[1].set_title(f'Distribution of Confidence')
    axes[1].set_xticks(bars)
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Proportion')
    axes[1].legend()

    # 显示图形
    plt.tight_layout()
    plt.legend()

    plt.show()


def get_CECE(label_test, pred_test, prob_test, nBin, num_classes, show_class_list):
    # 初始化字典，用于存储每个类别 ECE 值
    ECE_dict = {label_item: 0.0 for label_item in [lab for lab in range(num_classes)]}
    for c in range(num_classes):
        indices = np.where(label_test == c)[0]
        c_label = label_test[indices]
        c_pred = pred_test[indices]
        c_prob = prob_test[indices]

        bars, num_bars, acc_bars, conf_bars, ECE = get_ECE(c_prob, c_pred, c_label, nBin)
        if c in show_class_list:
            plt_ECE(bars, num_bars, acc_bars, len(c_label), ECE)

        ECE_dict[c] = ECE

    # 结果转列表
    ECE_list = list(ECE_dict.values())

    return ECE_list


def plt_CECE(ECE_list):
    # ECE 执行线性回归
    slope_ece, intercept_ece = np.polyfit(np.array([i for i in range(0, len(ECE_list))]), np.array(ECE_list), 1)
    Y_ece_fit = slope_ece * np.array([i for i in range(0, len(ECE_list))]) + intercept_ece

    # 创建绘图
    plt.figure(figsize=(8, 6))

    # 绘制图
    plt.scatter([i for i in range(0, len(ECE_list))], ECE_list, marker='^', color='#ff8066', label='ece')
    plt.plot([i for i in range(0, len(ECE_list))], Y_ece_fit, color='#c34a36', linestyle='--')
    plt.ylim([0, 1])

    # 添加标题和标签
    plt.title(f'Expected Calibration Error of Each Class, cece = {np.mean(ECE_list):.5f}')
    plt.xlabel('Class Labels')
    plt.ylabel('ECE')

    plt.tight_layout()
    plt.legend()
    # plt.savefig(f"类别校准-{dataset_name}-{method_name}.svg", format='svg')

    plt.show()


if __name__ == '__main__':
    file_path_test = f"../results/test/"

    # 读取测试集推理结果
    label_test = np.load(os.path.join(file_path_test, 'label_test.npy'))
    pred_test = np.load(os.path.join(file_path_test, 'pred_test.npy'))
    prob_test = np.load(os.path.join(file_path_test, 'prob_test.npy'))
    p = np.max(prob_test, axis=1)

    # 数据处理
    label_test = np.squeeze(label_test)
    pred_test = np.squeeze(pred_test)

    # 计算 CECE 所需数据
    nBin = 30
    show_class_list = []
    num_classes = prob_test.shape[1]
    ECE_list = get_CECE(label_test, pred_test, prob_test, nBin, num_classes, show_class_list)
    print("ECE of each class:", list(ECE_list))
    print("Average ECE of all classes:", np.mean(ECE_list))

    # 绘制 CECE　结果
    plt_CECE(ECE_list)

