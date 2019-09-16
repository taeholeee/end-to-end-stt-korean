import torch
import numpy as np
import re

class ZeroPadding(object):
    """Apply zero padding to tensor feature
    
    Args:
        pad_len (integer): max time step.
    """
    def __init__(self, pad_len=1536):
        assert isinstance(pad_len, int)
        self.pad_len = pad_len
    
    def __call__(self, tensor):
        if tensor is None:
            return None
        if tensor.dim() == 3:
            n_sample, n_time, n_feat = tensor.shape
            # result = F.pad(input=tensor, pad=(0, 0, 0, self.pad_len - n_time), value=0)
            padded = torch.zeros(n_sample, self.pad_len, n_feat)
            padded[:, :n_time, :] = tensor
        else:
            n_time, n_feat = tensor.shape
            padded = torch.zeros(self.pad_len, n_feat)
            padded[:n_time, :] = tensor
        return padded
    # Input x: list of np array with shape (timestep,feature)
# Return new_x : a np array of shape (len(x), padded_timestep, feature)

class Hangul(object):
    INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ") # char list: Hangul initials (초성)
    MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ") # char list: Hangul medials (중성)
    FINALS = list("∅ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ") # char list: Hangul finals (종성)
    SPACE_TOKEN = " "
    LABELS = sorted({SPACE_TOKEN}.union(INITIALS).union(MEDIALS).union(FINALS)) # char list: All CTC labels

    def check_syllable(self, char):
        return 0xAC00 <= ord(char) <= 0xD7A3

    def split_syllable(self, char):
        assert self.check_syllable(char)
        diff = ord(char) - 0xAC00
        _m = diff % 28
        _d = (diff - _m) // 28
        return (Hangul.INITIALS[_d // 21], Hangul.MEDIALS[_d % 21], Hangul.FINALS[_m])

    #   def preprocess(self, str):
    def __call__(self, str):
        if str is None:
            return None
        result = ""
        for char in re.sub("\\s+", Hangul.SPACE_TOKEN, str.strip()):
            if char == Hangul.SPACE_TOKEN:
                result += Hangul.SPACE_TOKEN
            elif self.check_syllable(char):
                result += "".join(self.split_syllable(char))
        return result

class Encoding(object):
    """Apply encoding to text label
    
    Args:
        pad_len (integer): max time step.
    """
    ctc_labels = Hangul.LABELS
    # labels = [" "] + ctc_labels
    labels = ctc_labels
    jamo2index = {k:(v+2) for v,k in enumerate(labels)}

    def __call__(self, string):
        if string is None:
            return None
        return list(map(lambda c: Encoding.jamo2index[c], string))


class OneHotEncoding(object):
    """Apply one hot encoding to numpy label
    
    Args:
        pad_len (integer): max time step.
    """
    def __init__(self, max_label_len=320, max_idx=55):
        assert isinstance(max_label_len, int)
        assert isinstance(max_idx, int)
        self.max_label_len = max_label_len
        self.max_idx = max_idx

    def __call__(self, Y):
        if Y is None:
            return None
        new_y = np.zeros((self.max_label_len, self.max_idx))
        new_y[0,0] = 1.0 # <sos>
        for idx, label in enumerate(Y):
            new_y[idx+1,label] = 1.0
        new_y[len(Y)+1,1] = 1.0 # <eos>
        return new_y


class Squeeze(object):
    """Squeeze tensor
    
    """    
    def __call__(self, tensor):
        if tensor is None:
            return None
        tensor.squeeze()
        return tensor

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if sample is None:
            return None
        return torch.from_numpy(sample)


class DimSizeFilter(object):
    """Filter tensor with specific size of dim.
    
    Args:
        dim (integer):       dimension of interest.
        max_size (integer) : size of filter
    """
    def __init__(self, dim, max_size):
        assert isinstance(dim, int)
        assert isinstance(max_size, int)
        self.dim = dim
        self.max_size = max_size

    def __call__(self, tensor):
        if tensor is None:
            return None
        tensor_size_of_interest = tensor.shape[self.dim]
        if tensor_size_of_interest > self.max_size:
            return None
        return tensor