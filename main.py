import os
import torch
import argparse
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from model import ImEDL
from dataset_reader import load_dataset_train, load_dataset_test, load_dataset_val
from loss_function import get_loss, get_device, compute_inter_centroid, compute_intra_trace, compute_difficulty, get_eta


def train_model(model, args):
    # 设置训练参数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    train_loss_list = []
    val_loss_list = []
    acc_train_list = []
    acc_val_list = []
    # ---------------------------------------------------- train ----------------------------------------------------
    space = " "
    best_val_acc = 0.0
    early_count = 0
    early_count_max = 10
    dcc = None
    xi = get_eta(args.data_name)
    for epoch in range(1, args.epochs + 1):
        # 分批次读取训练集
        train_loss = 0.0
        num_correct, num_sample = 0, 0
        list_evidence_src = []
        list_labels = []
        for i in range(args.k):
            train_loader = load_dataset_train(args.data_dir, args.data_name, i, args.batch_size)
            # 训练集训练
            model.train()
            for X, Y in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch:<9}"):
                # 模型向前传播，获得证据值
                X = X.to(device)
                L = Y.to(device)
                Y = torch.eye(args.num_classes)[Y].to(device)

                with torch.set_grad_enabled(True):
                    # 模型前传
                    evidence_src, alpha, w = model(X)

                    _, Y_pre = torch.max(alpha, dim=1)
                    Y_pre = torch.squeeze(Y_pre)
                    L = torch.squeeze(L)
                    num_correct += (Y_pre == L).sum().item()
                    num_sample += Y.shape[0]

                    # 反向传播
                    loss = get_loss(dcc=dcc, alpha=alpha, w=w, xi=xi, label=Y.float(),
                                    epoch_num=epoch, annealing_step=args.annealing_step, mu=args.mu,
                                    sigma=args.sigma, dataset_name=args.data_name)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    list_evidence_src.append(evidence_src.detach())
                    labels = torch.argmax(Y.float(), dim=1)
                    labels = labels.to(torch.int32)
                    list_labels.append(labels.detach())

        exp_lr_scheduler.step()

        acc_train = num_correct / num_sample
        acc_train_list.append(acc_train)
        print("acc_train = ", acc_train)

        avg_train_loss = train_loss / num_sample
        train_loss_list.append(avg_train_loss)
        print("train_loss = ", avg_train_loss)

        # 验证集验证
        # 分批次读取训练集
        val_loss = 0.0
        num_correct, num_sample = 0, 0
        model.eval()
        for i in range(args.k):
            val_loader = load_dataset_val(args.data_dir, args.data_name, i, args.batch_size)
            with torch.no_grad():
                for X, Y in tqdm(val_loader, total=len(val_loader), desc=f"Validation{space:<5}"):
                    X = X.to(device)
                    L = Y.to(device)
                    Y = torch.eye(args.num_classes)[Y].to(device)

                    evidence_src, alpha, w = model(X)

                    _, Y_pre = torch.max(alpha, dim=1)
                    Y_pre = torch.squeeze(Y_pre)
                    L = torch.squeeze(L)
                    num_correct += (Y_pre == L).sum().item()
                    num_sample += Y.shape[0]

                    loss = get_loss(dcc=dcc, alpha=alpha, w=w, xi=xi, label=Y.float(),
                                    epoch_num=epoch, annealing_step=args.annealing_step, mu=args.mu,
                                    sigma=args.sigma, dataset_name=args.data_name)

                    val_loss += loss.item()

        acc_val = num_correct / num_sample
        acc_val_list.append(acc_val)
        print("acc_val = ", acc_val)

        avg_val_loss = val_loss / num_sample
        val_loss_list.append(avg_val_loss)
        print("val_loss = ", avg_val_loss)

        # 更新 dcc
        total_evidence_src = torch.cat(list_evidence_src, dim=0).to(device)
        total_labels = torch.cat(list_labels, dim=0).to(device)
        intra_trace = compute_intra_trace(total_evidence_src.cpu().numpy(), total_labels.cpu().numpy())
        inter_centroid = compute_inter_centroid(total_evidence_src.cpu().numpy(), total_labels.cpu().numpy())
        dcc_numpy = compute_difficulty(intra_trace, inter_centroid)
        dcc = torch.from_numpy(dcc_numpy).float()

        # 更新 xi
        label_onehot = torch.eye(args.num_classes)[total_labels.cpu()]
        evidence_sum_src = torch.sum(total_evidence_src.cpu() * label_onehot, dim=0)
        evidence_mean_src = evidence_sum_src / label_onehot.sum(0).long()

        avg_evidence = evidence_mean_src
        xi = avg_evidence.max() / avg_evidence

        # 保存权重
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            early_count = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_best.pth'))
            print(f"New model saved!")
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_last.pth'))

        early_count += 1
        if early_count == early_count_max:
            print(f"Early Stopping! count = {early_count}")
            break

    # 保存损失和正确率
    np.save(os.path.join(args.save_dir, 'train_loss.npy'), np.array(train_loss_list))
    np.save(os.path.join(args.save_dir, 'val_loss.npy'), np.array(val_loss_list))
    np.save(os.path.join(args.save_dir, 'acc_train.npy'), np.array(acc_train_list))
    np.save(os.path.join(args.save_dir, 'acc_val.npy'), np.array(acc_val_list))


def test(model, args):
    # 模型权重加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.weights_dir, f'model_best.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 设置测试设备
    model.to(device)

    # 分批次读取训练集
    alpha_all, uncertainty_all, prob_all, pred_all, label_all = [], [], [], [], []
    for i in range(args.k):
        test_loader = load_dataset_test(args.data_dir, args.data_name, i, 1)
        alpha_test, uncertainty_test, prob_test, pred_test, label_test = eval_one(model, test_loader)
        label_test = np.squeeze(label_test)
        pred_test = np.squeeze(pred_test)
        alpha_all.append(alpha_test)
        uncertainty_all.append(uncertainty_test)
        prob_all.append(prob_test)
        pred_all.append(pred_test)
        label_all.append(label_test)

    alpha_all = np.concatenate(alpha_all, axis=0)
    uncertainty_all = np.concatenate(uncertainty_all, axis=0)
    prob_all = np.concatenate(prob_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    # 保存结果
    dir_result = os.path.join(args.save_dir, "test")
    os.makedirs(dir_result, exist_ok=True)
    np.save(os.path.join(dir_result, 'alpha_test.npy'), alpha_all)
    np.save(os.path.join(dir_result, 'uncertainty_test.npy'), uncertainty_all)
    np.save(os.path.join(dir_result, 'prob_test.npy'), prob_all)
    np.save(os.path.join(dir_result, 'pred_test.npy'), pred_all)
    np.save(os.path.join(dir_result, 'label_test.npy'), label_all)

    # 计算分类正确率
    is_correct = label_all == pred_all
    accuracy = np.mean(is_correct)
    print(f"acc_test  : {accuracy * 100:.2f}%")


def test_ood(model, args):
    # 模型权重加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.weights_dir, f'model_best.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 设置测试设备
    model.to(device)

    # 分批次读取训练集
    alpha_all, uncertainty_all, prob_all, pred_all, label_all = [], [], [], [], []
    for i in range(args.k):
        test_loader = load_dataset_test(args.data_dir, args.data_name, i, 1)
        alpha_test, uncertainty_test, prob_test, pred_test, label_test = eval_one(model, test_loader)
        label_test = np.squeeze(label_test)
        pred_test = np.squeeze(pred_test)
        alpha_all.append(alpha_test)
        uncertainty_all.append(uncertainty_test)
        prob_all.append(prob_test)
        pred_all.append(pred_test)
        label_all.append(label_test)

    alpha_all = np.concatenate(alpha_all, axis=0)
    uncertainty_all = np.concatenate(uncertainty_all, axis=0)
    prob_all = np.concatenate(prob_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    # 保存结果
    dir_result = os.path.join(args.save_dir, "ood")
    os.makedirs(dir_result, exist_ok=True)
    np.save(os.path.join(dir_result, 'alpha_test.npy'), alpha_all)
    np.save(os.path.join(dir_result, 'uncertainty_test.npy'), uncertainty_all)
    np.save(os.path.join(dir_result, 'prob_test.npy'), prob_all)
    np.save(os.path.join(dir_result, 'pred_test.npy'), pred_all)
    np.save(os.path.join(dir_result, 'label_test.npy'), label_all)

    # 计算分类正确率
    is_correct = label_all == pred_all
    accuracy = np.mean(is_correct)
    print(f"acc_test  : {accuracy * 100:.2f}%")


def eval_one(model, data_loader):
    device = get_device()

    model.eval()
    space = " "
    alpha, label = [], []
    for X, Y in tqdm(data_loader, total=len(data_loader), desc=f"Test{space:<11}"):
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            _, a, _ = model(X)

            a = a.squeeze()
            a = a.detach().cpu().numpy()
            Y = Y.squeeze()
            Y = Y.detach().cpu().numpy()

            alpha.append(a)
            label.append(Y)

    alpha = np.array(alpha)
    label = np.array(label)

    # 计算 pred, prob, uncertainty
    num_classes = args.num_classes
    num_samples = len(label)

    uncertainty, prob, pred = [], [], []
    for n in range(num_samples):
        item = alpha[n]
        uncertainty.append(num_classes / np.sum(item))
        prob.append(item / np.sum(item))
        pred.append(np.argmax(item))

    uncertainty = np.array(uncertainty)
    prob = np.array(prob)
    pred = np.array(pred)

    return alpha, uncertainty, prob, pred, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=200, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--annealing_step', type=int, default=10, help='increase the value of lambda')
    parser.add_argument('--k', type=int, default=1, help='k')

    parser.add_argument('--data_name', type=str, default="mnist", help='name of dataset')
    parser.add_argument('--data_dir', type=str, default="./datasets/mnist/imbalance/",
                        help='dir of dataset')
    parser.add_argument('--save_dir', type=str, default="./results/",
                        help='dir of results')
    parser.add_argument('--weights_dir', type=str, default="./results/",
                        help='dir of weights')

    parser.add_argument('--channels', type=int, default=1, help='channels')
    parser.add_argument('--num_classes', type=int, default=10, help='classes')
    parser.add_argument('--mem_h', type=int, default=10, help='hyper parameters h')
    parser.add_argument('--mu', type=float, default=0.01, help='hyper parameters mu')
    parser.add_argument('--sigma', type=float, default=10, help='hyper parameters sigma')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')

    args = parser.parse_args()

    model = ImEDL(args.channels, args.num_classes, args.mem_h)

    train_model(model, args)

    test(model, args)
    # test_ood(model, args)
