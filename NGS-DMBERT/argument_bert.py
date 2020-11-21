from __future__ import print_function
from constant import *
from models import *
from dataset import *
from utils import f_score
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import f1_score,precision_score,recall_score

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0 

TrainSet = Dataset("train")
DevelopSet = Dataset("develop")
TestSet = Dataset("test")
acc_NA = Accuracy()
acc_not_NA = Accuracy()
acc_total = Accuracy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda:
    torch.cuda.set_device(device)

discriminator = Discriminator_argument()
discriminator.to(device, non_blocking=True)
param_optimizer=list(discriminator.named_parameters())
no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
optimizer_grouped_parameters=[
    {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
    {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
]
opt=BertAdam(optimizer_grouped_parameters,lr=Lr,warmup=0.1,t_total=Epoch*TrainSet.AtrainNum)
bst=0.0
e_bst=0

def test(e, dataset):
    if dataset == TestSet:
       discriminator.load_state_dict(torch.load("Amodel_bert.tar"))
    discriminator.eval()
    preds=[]
    labels=[]
    pred_subtypes=[]
    golden_subtypes=[]

    for words, inMask, maskL, maskM, maskR, label, subtype, subtype_pred in dataset.batchs_arg():
        words, inMask, maskL, maskM, maskR, label = words.to(device), inMask.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device)
        loss, scores, pred = discriminator(words, inMask, maskL, maskM, maskR, label)
        preds.append(pred[1].cpu().numpy())
        labels.append(label.cpu().numpy())
        pred_subtypes.extend(subtype_pred)
        golden_subtypes.extend(subtype)
    cnt=0
    cnt1=0
    FN=0
    FP=0
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)
    pred_subtypes=np.array(pred_subtypes)
    golden_subtypes=np.array(golden_subtypes)

    if dataset == TestSet or True:
        for i in range(0,preds.shape[0]):
            if labels[i]==0:
                cnt1+=1
            if preds[i]!=labels[i]:
                cnt+=1
                if preds[i]==0 and labels[i]!=0:
                    FN+=1
                if preds[i]!=0 and labels[i]==0:
                    FP+=1
        print("EVAL %s #Wrong %d #NegToPos %d #PosToNeg %d #All %d #Negs %d"%("Test",cnt,FP,FN,len(preds),cnt1))
    acc, _, f1 = f_score(preds, labels, pred_subtypes, golden_subtypes)
    print ("acc:{}, f1:{}".format(acc,f1))
    global bst, e_bst
    if f1 > bst:
        bst = f1
        e_bst = e 
        torch.save(discriminator.state_dict(), "Amodel_bert.tar")
    print("BST: %f, epoch: %d"%(bst, e_bst)) 
    return f1
 
def oneEpoch(e, dataset):
    print ("Epoch %d"%e)
    cnt = 0
    acc_NA.clear()
    acc_not_NA.clear()
    acc_total.clear()

    setIter = dataset.batchs_arg()
    for words, inMask, maskL, maskM, maskR, label in setIter:
        opt.zero_grad()
        words, inMask, maskL, maskM, maskR, label = words.to(device), inMask.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device)
        loss, socres, preds = discriminator(words, inMask, maskL, maskM, maskR, label)
        loss.backward()
        opt.step()
        for i, pred in enumerate(preds[1].data):
            if label[i] == 0:
                acc_NA.add(pred == label[i])
            else:
                acc_not_NA.add(pred == label[i])
            acc_total.add(pred == label[i])
        time_str = datetime.datetime.now().isoformat()
        if cnt % 100 == 0:
            sys.stdout.write("%s step %d | loss: %f, acc_not_NA: %f, acc_NA: %f, acc_total: %f\r" % (time_str, cnt, loss, acc_not_NA.get(), acc_NA.get(), acc_total.get()))
            sys.stdout.flush()
        cnt+=1

def train():
    discriminator.train()
    for e in range(0, Epoch):
        discriminator.train()
        oneEpoch(e, TrainSet)
        test(e, TestSet)
    test(e, TestSet)

if __name__=='__main__':
    train()