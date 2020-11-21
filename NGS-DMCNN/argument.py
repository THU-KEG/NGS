from __future__ import print_function
from constant import *
from models import *
from dataset import *
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score,precision_score,recall_score
from utils import f_score

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

discriminator = Discriminator_argument()
discriminator.to(device, non_blocking=True)
opt = optim.Adadelta(discriminator.parameters(), lr=Lr, rho=0.95, eps=1e-06)
bst=0.0

def test(e, dataset):
    if dataset == TestSet:
        discriminator.load_state_dict(torch.load("Amodel.tar"))
    discriminator.eval()
    preds=[]
    labels=[]
    pred_subtypes=[]
    golden_subtypes=[]

    for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, subtype_golden in dataset.batchs_arg():
        words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label = words.to(device), pos1.to(device), pos2.to(device), loc.to(device), loc_mark.to(device), subtype.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device)
        loss, scores, pred = discriminator(words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label)
        preds.append(pred[1].cpu().numpy())
        labels.append(label.cpu().numpy())
        pred_subtypes.append(subtype.cpu().numpy())
        golden_subtypes.append(subtype_golden.cpu().numpy())
    cnt=0
    cnt1=0
    FN=0
    FP=0
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)
    pred_subtypes=np.concatenate(pred_subtypes,0)
    golden_subtypes=np.concatenate(golden_subtypes,0)
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
    global bst
    if f1 > bst and dataset == DevelopSet:
        print("BST: %f, epoch: %d"%(bst, e))
        torch.save(discriminator.state_dict(), "Amodel.tar")
        bst = f1  
    elif dataset == TestSet:
        print ("Test acc: %f, F1: %f"%(acc, f1))
    return f1
 
def oneEpoch(e, dataset):
    print ("Epoch %d"%e)
    cnt = 0
    acc_NA.clear()
    acc_not_NA.clear()
    acc_total.clear()

    setIter = dataset.batchs_arg()
    for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, subtype_golden in setIter:
        opt.zero_grad()
        words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label = words.to(device), pos1.to(device), pos2.to(device), loc.to(device), loc_mark.to(device), subtype.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device)
        loss, socres, preds = discriminator(words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label)
        loss.backward()
        opt.step()
        for i, pred in enumerate(preds[1].data):
            if label[i] == 0:
                acc_NA.add(pred == label[i])
            else:
                acc_not_NA.add(pred == label[i])
            acc_total.add(pred == label[i])
        if cnt % 50  == 0:
            sys.stdout.write("epoch %d step %d | loss: %f, acc_NA: %f, acc_not_NA: %f, acc_total: %f\r" % (e, cnt, loss, acc_NA.get(), acc_not_NA.get(), acc_total.get()))
            sys.stdout.flush()
        cnt+=1
    
def train():
    discriminator.train()
    for e in range(0, Epoch):
        discriminator.train()
        oneEpoch(e, TrainSet)
        test(e, DevelopSet)
    test(e, TestSet)
    infer("train")
    infer("develop")
    infer("test")

def infer(Tag):
    dataset=Dataset(Tag)
    discriminator.load_state_dict(torch.load("Amodel.tar"))
    discriminator.eval()
    preds = []
    labels = []
    pred_subtypes=[]
    golden_subtypes=[]
    senLabels = []
    ettIdxs = []
    ettLengths = []
    results = []
    for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel, subtype_golden in dataset.batchs_gibbs():
        words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel = words.to(device), pos1.to(device), pos2.to(device), loc.to(device), loc_mark.to(device), subtype.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device), ettIdx.to(device), ettLength.to(device), senLabel.to(device)
        loss, scores, pred = discriminator(words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label)
        preds.extend(pred[1].cpu().numpy())
        labels.extend(label.cpu().numpy())
        senLabels.extend(senLabel.cpu().numpy().tolist())
        ettIdxs.extend(ettIdx)
        ettLengths.extend(ettLength)
        for i in range(len(words)):
            state = {}
            results.append(state)
    argRoles = []
    def gen_argRole(pred, ettIdx, ettLength):
        for j in range(len(pred)):
            argRole = np.zeros((SenLen), dtype = np.int64)
            for arg_idx in range(len(pred)):
                arg_in_sen = list(range(ettIdx[arg_idx], ettIdx[arg_idx] + ettLength[arg_idx]))
                if arg_idx != j:        
                    if pred[arg_idx] == 0:
                        argRole[arg_in_sen] = Ett_tag
                    else:
                        argRole[arg_in_sen] = pred[arg_idx]
                else:
                    argRole[arg_in_sen] = predEtt_tag
            argRoles.append(argRole)
    L = 0
    R = 0
    sen_idx = senLabels[0]
    for i in range(len(senLabels)):
        if senLabels[i] == sen_idx:
                R += 1
        else:
            sen_idx = senLabels[i]
            if L != R:
                gen_argRole(preds[L:R], ettIdxs[L:R], ettLengths[L:R])
                L = R
                R += 1
                sen_idx = senLabels[i]
    if L<len(preds):
        gen_argRole(preds[L:], ettIdxs[L:], ettLengths[L:])
    np.save(dataPath + Tag + "_argRole_arg_noise.npy", np.array(argRoles))

if __name__=='__main__':
    train()