import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import matplotlib.pyplot as plt
import argparse
from utils import DataLoader, CustomDataset
from model import RegressionNet

def main(args):
    # specify the saving path
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    save_path = "./weights/"

    # hyper parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LR = args.lr
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    VERBOSE = args.verbose
    VALID = args.valid
    SPLIT_RATIO = args.ratio

    # load dataset
    dataset = CustomDataset()
    dataset.read_csv("data.csv", header=None)
    train_data, test_data = dataset.train_test_split(SPLIT_RATIO)

    # use dataloader for iteratively batch-training
    train_dataloader = DataLoader(train_data, BATCH_SIZE, data_dim=5, shuffle=True, seed=0)
    test_dataloader = DataLoader(test_data, 1, data_dim=5, shuffle=False)

    # build model
    model = RegressionNet(5, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # validation before training
    if VALID:
        print("[INFO] preparing visualizing tool")
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        ax1 = fig1.add_subplot(111)
        ax1.set_title("Loss")
        ax2 = fig2.add_subplot(221)
        ax2.set_title("PWM1")
        ax3 = fig2.add_subplot(222)
        ax3.set_title("PWM2")
        ax4 = fig2.add_subplot(223)
        ax4.set_title("PWM3")
        ax5 = fig2.add_subplot(224)
        ax5.set_title("PWM4")
        with torch.no_grad():
            x, y, y_hat = [], [], []
            for it, batch in enumerate(test_dataloader):
                data, target = batch
                data = data.to(device)
                target = target.to(device)

                pred = model(data)

                x.append(it)
                y.append(target.cpu().numpy())
                y_hat.append(pred.cpu().numpy())

            x, y, y_hat = np.array(x), np.array(y), np.array(y_hat)

            ax2.plot(y.squeeze()[:, 0], color="red", label="Ground Truth")
            ax2.plot(y_hat.squeeze()[:, 0], color="blue", label="Predicted (Before Training)")
            ax3.plot(y.squeeze()[:, 1], color="red")
            ax3.plot(y_hat.squeeze()[:, 1], color="blue")
            ax4.plot(y.squeeze()[:, 2], color="red")
            ax4.plot(y_hat.squeeze()[:, 2], color="blue")
            ax5.plot(y.squeeze()[:, 3], color="red")
            ax5.plot(y_hat.squeeze()[:, 3], color="blue")

    # start training
    epoch_loss = []
    print("\n[INFO] start training")
    total_num = train_data.shape[0]
    for e in range(EPOCHS):
        loss_ls = []
        if VERBOSE:
            progress = 0.0
        for it, batch in enumerate(train_dataloader):
            data, target = batch
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if VERBOSE:
                progress += (data.shape[0] / total_num) * 100
                print(f"Epoch: {e}, Iter {it}, Progress: {progress:.1f}%, Loss: {loss.data:.6f}")

            loss_ls.append(loss.data)
        epoch_loss.append(np.array(loss_ls).sum())

    # validation after training
    if VALID:
        print("\n[INFO] start validation")
        with torch.no_grad():
            x, y, y_hat = [], [], []
            for it, batch in enumerate(test_dataloader):
                data, target = batch
                data = data.to(device)
                target = target.to(device)

                pred = model(data)

                x.append(it)
                y.append(target.cpu().numpy())
                y_hat.append(pred.cpu().numpy())

            x, y, y_hat = np.array(x), np.array(y), np.array(y_hat)

            ax2.plot(y_hat.squeeze()[:, 0], color="green", label="Predicted (After Training)")
            ax3.plot(y_hat.squeeze()[:, 1], color="green")
            ax4.plot(y_hat.squeeze()[:, 2], color="green")
            ax5.plot(y_hat.squeeze()[:, 3], color="green")

    # save training results
    print("\n[INFO] save results")
    file_name = save_path + "estimator.ckpt"
    torch.save(model.state_dict(), file_name)

    # visualize the comparing result
    if VALID:
        print("\n[INFO] visualize comparison")
        ax1.plot(epoch_loss, color="red")
        ax1.set_xlabel("Epoch Num")
        ax1.set_ylabel("Mean Square Error")
        fig2.tight_layout(rect=[0,0.15,1,1])
        fig2.legend(loc="lower right")
        fig1.savefig("loss.png")
        fig2.savefig("compare.png")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--batch", type=int, default=1024, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="total epochs")
    parser.add_argument("--verbose", action="store_true", help="display training message")
    parser.add_argument("--valid", action="store_true", help="plot validation result")
    parser.add_argument("--ratio", type=float, default=0.999, help="specify how many data will be used")
    args = parser.parse_args()

    main(args)
