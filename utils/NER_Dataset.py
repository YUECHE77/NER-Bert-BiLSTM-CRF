from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, dataset):
        super(NerDataset, self).__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
