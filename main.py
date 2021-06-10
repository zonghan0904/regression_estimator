import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import matplotlib.pyplot as plt
import argparse
from utils import DataLoader
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

    # load dataset
    x = np.random.randn(10000, 1).astype(np.float32)
    y = 3 * x + 1
    vx = np.random.randn(10, 1).astype(np.float32)
    vy = 3 * vx + 1
    dataset = np.concatenate((x, y), axis=1)

    # use dataloader for iteratively batch-training
    dataloader = DataLoader(dataset, BATCH_SIZE, data_dim=1, shuffle=True, seed=0)

    # build model
    model = RegressionNet(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # validation before training
    print("[INFO] preparing visualizing tool")
    fig = plt.figure()
    with torch.no_grad():
        validation_data = torch.FloatTensor(vx).to(device)
        pred = model(validation_data).cpu().detach().numpy()

        ax = fig.add_subplot(121)
        ax.plot(vx.reshape(-1), vy.reshape(-1), "ro", label="Ground Truth")
        ax.plot(vx.reshape(-1), pred.reshape(-1), "b^", label="Predicted")
        ax.legend()
        ax.set_title("Before Training")

    # start training
    print("\n[INFO] start training")
    total_num = dataset.shape[0]
    for e in range(EPOCHS):
        if VERBOSE:
            progress = 0.0
        for it, batch in enumerate(dataloader):
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

    # validation after training
    print("\n[INFO] start validation")
    with torch.no_grad():
        validation_data = torch.FloatTensor(vx).to(device)
        pred = model(validation_data).cpu().detach().numpy()

        ax = fig.add_subplot(122)
        ax.plot(vx.reshape(-1), vy.reshape(-1), "ro", label="Ground Truth")
        ax.plot(vx.reshape(-1), pred.reshape(-1), "b^", label="Predicted")
        ax.set_title("After Training")
        ax.legend()

    # save training results
    print("\n[INFO] save results")
    file_name = save_path + "estimator.ckpt"
    torch.save(model.state_dict(), file_name)
    plt.savefig("compare.png")

    # visualize the comparing result
    print("\n[INFO] visualize comparison")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--batch", type=int, default=1048576, help="batch size")
    parser.add_argument("--epoch", type=int, default=100, help="total epochs")
    parser.add_argument("--verbose", action="store_true", help="display training message")
    args = parser.parse_args()

    main(args)
