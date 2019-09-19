import torch.nn.functional as F
import torch.nn as nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)
    
def label_smoothing_loss(pred_y,true_y,label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    criterion = nn.NLLLoss(ignore_index=0)

    if label_smoothing == 0.0:
        pred_y = pred_y.permute(0,2,1)#pred_y.contiguous().view(-1,output_class_dim)
        # true_y = torch.max(true_y,dim=2)[1][:,:max_label_len].contiguous()#.view(-1)
        true_y = torch.max(true_y,dim=2)[1].contiguous()#.view(-1)
        loss = criterion(pred_y, true_y)
    else:
        # true_y = true_y[:,:max_label_len,:].contiguous()
        true_y = true_y.contiguous()
        # true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        seq_len = torch.sum(torch.sum(true_y,dim=-1),dim=-1,keepdim=True)
    
        # calculate smoothen label, last term ensures padding vector remains all zero
        class_dim = true_y.size()[-1]
        smooth_y = ((1.0-label_smoothing)*true_y+(label_smoothing/class_dim))*torch.sum(true_y,dim=-1,keepdim=True)

        loss = - torch.mean(torch.sum((torch.sum(smooth_y * pred_y,dim=-1)/seq_len),dim=-1))

    return loss