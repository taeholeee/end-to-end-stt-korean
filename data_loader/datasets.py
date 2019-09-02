import os
import torch
from torch.utils.data import Dataset, DataLoader
import kaldiio
from torchvision import transforms

class KaldiDataset(Dataset):
    """
    Kaldi dataset using kaldiio
    """
    def __init__(self, feats_file, labels_file=None, root_dir=None, transform=None, target_transform=None):
        """
        Args:
            feats_file (string): 
            labels_file (string):
            root_dir (string):
        """
        self.feats = kaldiio.load_scp( os.path.join(root_dir, feats_file) )
        if labels_file is not None:
            self.labels = self._load_text( os.path.join(root_dir, labels_file) )
        else:
            self.labels = None
        self.utts = list(self.feats.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.utts)
    
    def __getitem__(self, idx):
        utt_name = self.utts[idx]
        feat = self.feats[utt_name]

        if self.transform is not None:
            feat = self.transform(feat)

        if self.labels is not None:
            label = self.labels[utt_name]
            if self.target_transform is not None:
                label = self.target_transform(label)
            return feat, label
        else:
            return feat

    def _load_text(self, fname, separator=None):
        """Lazy loader for kaldi text file.

        Args:
            fname (str or file(text mode)):
            separator (str):
        """
        load_func = str
        loader = kaldiio.utils.LazyLoader(load_func)
        with open(fname, 'r') as fd:
            for line in fd:
                seps = line.split(separator, 1)
                if len(seps) != 2:
                    raise ValueError(
                        'Invalid line is found:\n>   {}'.format(line))
                token, arkname = seps
                loader[token] = arkname.rstrip()
        return loader
