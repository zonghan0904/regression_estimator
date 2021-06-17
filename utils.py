import numpy as np
import torch
import pandas

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

class CustomDataset():
    def __init__(self):
        self.data = None
        self._shape = None
        self.empty = True

    def read_csv(self, filename, header=None):
        self.data = pandas.read_csv(filename, header=header).to_numpy()
        self.empty = False

    def train_test_split(self, ratio):
        assert not self.empty

        size = self.__len__()
        ratio = max(min(ratio, 1.0), 0.0)
        train_num = int(size * ratio)
        test_num = size - train_num
        randomstate = np.random.default_rng(None)
        idx = randomstate.choice(np.arange(0, size), size, replace=False)
        tmp_data = self.data[idx]

        train_data = tmp_data[:train_num]
        test_data = None if test_num < 0 else tmp_data[train_num:]

        return train_data, test_data

    @property
    def shape(self):
        self._shape = self.data.shape if not self.empty else None
        return self._shape

    def __len__(self):
        l = self.shape[0] if not self.empty else 0
        return l

    def __str__(self):
        if self.empty:
            s = "<Empty Dataset>"
        else:
            s = f"<{self.data}, Shape {self.shape}>"
        return s


if __name__ == "__main__":
    dataset = CustomDataset()
    dataset.read_csv("data.csv", header=None)
    train_data, test_data = dataset.train_test_split(0.99)
    print(train_data.shape)

    BATCH_SIZE = 1

    dataloader = DataLoader(train_data, BATCH_SIZE, 5)
    print(dataloader)
    print()

    for e in range(2):
        for i, batch in enumerate(dataloader):
            x, y = batch
            print(f"{e} epoch, {i} iter, data: {x}, target:{y}")

