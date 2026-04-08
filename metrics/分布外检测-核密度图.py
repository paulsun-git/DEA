import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    file_path_test = f'../results/test/'
    file_path_ood = f'../results/ood/'

    un_test = np.load(os.path.join(file_path_test, 'uncertainty_test.npy'))
    un_ood = np.load(os.path.join(file_path_ood, 'uncertainty_test.npy'))
    # ---------------------------------------------- 核密度估计 ----------------------------------------------
    plt.figure(figsize=(5, 4), dpi=300)

    # 绘制三个列表的核密度估计
    sns.kdeplot(un_test, label='test', fill=True)
    sns.kdeplot(un_ood, label='OOD', fill=True)
    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')

    # 显示图形
    # plt.savefig(f"核密度图-{dataset_name}-{method_name}.svg", format='svg')
    plt.show()


