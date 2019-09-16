import torch
import editdistance as ed

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# letter_error_rate function
# Merge the repeated prediction and calculate editdistance of prediction and ground truth
def letter_error_rate(pred_y,true_y):
    pred_y = torch.max(pred_y,dim=2)[1].cpu().numpy()
    true_y = torch.max(true_y,dim=2)[1].cpu().data.numpy()
    ed_accumalate = []
    for p,t in zip(pred_y,true_y):
        compressed_t = [w for w in t if (w!=1 and w!=0)]
        
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        # if data == 'timit':
        #     compressed_t = collapse_phn(compressed_t)
        #     compressed_p = collapse_phn(compressed_p)
        try:
            ed_accumalate.append(ed.eval(compressed_p,compressed_t)/len(compressed_t))
        except ZeroDivisionError:
            ed_accumalate.append(1.0)
    return sum(ed_accumalate)/len(ed_accumalate)