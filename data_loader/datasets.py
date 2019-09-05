import os
import torch
from torch.utils.data import Dataset, DataLoader
import kaldiio
from .transforms import Hangul

class KaldiDataset(Dataset):
    """
    Kaldi dataset using kaldiio
    """
    def __init__(self, feats_file, labels_file=None, root_dir=None, transform=None, target_transform=None, max_timestep=1024, max_label_len=128):
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
        self.max_timestep = max_timestep
        self.max_label_len = max_label_len
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _safe_indices and similarly for unsafe
        # samples.
        self._safe_indices = []
        self._unsafe_indices = []

    def _safe_get_item(self, idx):
        """Returns None instead of throwing an error when dealing with an
        unsafe sample, and also builds an index of safe and unsafe samples as
        and when they get accessed.
        """
        # differentiates IndexError occuring here from one occuring during
        # sample loading
        utt_name = self.utts[idx]
        feat = self.feats[utt_name]
        convert_hangul = Hangul()
        if idx in self._safe_indices:               # when idx is aleady checked and it is safe
            if self.labels is not None:
                label = self.labels[utt_name]
                return feat, label
            else:
                return feat
        
        elif idx in self._unsafe_indices:           # when idx is aleady checked but it is unsafe
            feat = None
            if self.labels is not None:
                label = None
                return feat, label
            else:
                return feat

        else:                                       # when idx is not checked
            if feat.shape[0] > self.max_timestep:
                feat =  None
                self._unsafe_indices.append(idx)

            if self.labels is not None:
                label = self.labels[utt_name]
                syllables = convert_hangul(label)
                if len(syllables) > self.max_label_len:
                    label = None
                    if idx not in self._unsafe_indices:
                        self._unsafe_indices.append(idx)
                elif feat is not None:
                    self._safe_indices.append(idx)
                return feat, label
            else:
                if feat is not None:
                    self._safe_indices.append(idx)
                return feat

    def __len__(self):
        return len(self.utts)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            feat, label = self._safe_get_item(idx)
            if feat is not None and label is not None:
                feat = self.transform(feat)
                label = self.target_transform(label)
                return feat, label
            else:
                return None
        else:
            feat = self._safe_get_item(idx)
            if feat is not None:
                feat = self.transform(feat)
                return feat
            else:
                return None


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
