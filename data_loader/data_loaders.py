from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.transforms import ZeroPadding, Hangul, Encoding, OneHotEncoding, Squeeze
from data_loader.datasets import KaldiDataset

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
    def __init__(self, feats_file, labels_file=None, root_dir=None, batch_size=64, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = transforms.Compose([
            transforms.ToTensor(), 
            ZeroPadding()
        ])
        target_trsfm = transforms.Compose([
            Hangul(), 
            Encoding(), 
            OneHotEncoding(), 
            transforms.ToTensor()
        ])
        self.feats_file = feats_file
        self.labels_file = labels_file
        self.root_dir = root_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=tsfrm)
        self.dataset = KaldiDataset(self.feats_file, self.labels_file, self.root_dir, transform=trsfm, target_transform=target_trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

