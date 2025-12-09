from torch import tensor
from torch.utils.data import Dataset


class SpecDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spec = self.dataset[idx]['spectrum']
        # 将列表转换为张量
        spec = spec.clone().detach().unsqueeze(-1)[:3900,:]

        # print('spec = ', spec)
        # print('spec.shape = ', spec.shape)
        return {
            "spectrum": spec
        }