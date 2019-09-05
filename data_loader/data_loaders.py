from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.transforms import ZeroPadding, Hangul, Encoding, OneHotEncoding, Squeeze, ToTensor
from data_loader.datasets import KaldiDataset
from torchvision import transforms
# import nonechucks as nc
from torch.utils.data.dataloader import default_collate

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class KaldiDataLoader(BaseDataLoader):
    """
    Kaldi data loading demo using BaseDataLoader
    """
    def __init__(self, feats_file, labels_file=None, root_dir=None, batch_size=64, max_timestep=1024, max_label_len=128, output_class_dim=55, shuffle=True, validation_split=0.0, num_workers=1):

        self.feats_file = feats_file
        self.labels_file = labels_file
        self.root_dir = root_dir
        self.max_timestep = max_timestep
        self.max_label_len = max_label_len
        self.output_class_dim = output_class_dim
        trsfm = transforms.Compose([
            ToTensor(), 
            ZeroPadding(pad_len=self.max_timestep)
        ])
        target_trsfm = transforms.Compose([
            Hangul(), 
            Encoding(), 
            OneHotEncoding(max_label_len=self.max_label_len, max_idx=self.output_class_dim), 
            ToTensor()
        ])
        self.dataset = KaldiDataset(self.feats_file, self.labels_file, self.root_dir, transform=trsfm, target_transform=target_trsfm, max_timestep=self.max_timestep, max_label_len=self.max_label_len)
        # self.dataset = nc.SafeDataset(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self._collate_with_filter)

    def _collate_with_filter(self, batch):
        """Put each data field into a tensor with outer dimension batch size
        """
        batch = list(filter( lambda x:x is not None, batch))
        return default_collate(batch)

