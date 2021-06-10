import numpy as np
import torch

class DataLoader():
    def __init__(self, dataset, batch_size, data_dim, shuffle=True, seed=None):
        self.dataset = dataset
        self.max_size = dataset.shape[0]
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.randomstate = np.random.default_rng(seed)
            self.idx = self.randomstate.choice(np.arange(0, self.max_size), self.max_size, replace=False)
        else:
            self.idx = np.arange(0, self.max_size)

    def sample_once(self, batch_size):
        batch_size = self.max_size if batch_size > self.max_size else batch_size
        idx = np.random.randint(0, self.max_size, size=batch_size)
        return (torch.FloatTensor(self.dataset[idx, :self.data_dim]),
                torch.FloatTensor(self.dataset[idx, self.data_dim:]))

    def __iter__(self):
        batch = []
        for i in range(self.max_size):
            batch.append(self.dataset[self.idx[i]])
            if len(batch) >= self.batch_size:
                batch = torch.FloatTensor(batch)
                yield batch[:, :self.data_dim], batch[:, self.data_dim:]
                batch = []
        if len(batch) > 0:
            batch = torch.FloatTensor(batch)
            yield batch[:, :self.data_dim], batch[:, self.data_dim:]
        if self.shuffle:
            self.idx = self.randomstate.choice(np.arange(0, self.max_size), self.max_size, replace=False)

    def __len__(self):
        return self.max_size

    def __str__(self):
        s = f"<batch size: {self.batch_size}, data dim: {self.data_dim}, shuffle: {self.shuffle}"
        s += f", seed: {self.seed}\ndataset:\n{self.dataset}"
        s += ", DataLoader Object>"
        return s

if __name__ == "__main__":
    BATCH_SIZE = 1

    dataset = np.arange(0, 10).reshape(-1, 2)

    dataloader = DataLoader(dataset, BATCH_SIZE, 1)
    print(dataloader)
    print()

    for e in range(2):
        for i, batch in enumerate(dataloader):
            x, y = batch
            print(f"{e} epoch, {i} iter, data: {x}, target:{y}")

