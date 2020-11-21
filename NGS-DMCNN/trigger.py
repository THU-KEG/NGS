from __future__ import print_function
from constant import *
from models import *
from dataset import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

discriminator = Discriminator()
discriminator.to(device, non_blocking=True)
opt = optim.Adadelta(discriminator.parameters(), lr=Lr, rho=0.95, eps=1e-06)
bst=0.0

def test(e, dataset):
    if dataset == TestSet:
        discriminator.load_state_dict(torch.load("Tmodel.tar"))
    discriminator.eval()
    preds=[]
    labels=[]

    for words, pos, loc, maskL, maskR, label in dataset.batchs():
        words, pos, loc, maskL, maskR, label = words.to(device), pos.to(device), loc.to(device), maskL.to(device), maskR.to(device), label.to(device)
        loss, scores, pred = discriminator(words, pos, loc, maskL, maskR, label)
        preds.append(pred[1].cpu().numpy())
        labels.append(label.cpu().numpy())
    cnt=0
    cnt1=0
    FN=0
    FP=0
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)

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
    acc = precision_score(labels, preds,labels = list(range(1, 34)), average = "micro")
    f1 = f1_score(labels, preds, labels = list(range(1, 34)), average = "micro")
    print ("acc:{}, f1:{}".format(acc,f1))
    global bst
    if f1 > bst and dataset == DevelopSet:
        print("BST: %f, epoch: %d"%(bst, e))
        torch.save(discriminator.state_dict(), "Tmodel.tar")
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

    setIter = dataset.batchs()
    for words, pos, loc, maskL, maskR, label in setIter:
        opt.zero_grad()
        words, pos, loc, maskL, maskR, label = words.to(device), pos.to(device), loc.to(device), maskL.to(device), maskR.to(device), label.to(device)
        loss, scores, preds = discriminator(words, pos, loc, maskL, maskR, label)
        loss.backward()
        opt.step()
        for i, pred in enumerate(preds[1].data):
            if label[i] == 0:
                acc_NA.add(pred == label[i])
            else:
                acc_not_NA.add(pred == label[i])
            acc_total.add(pred == label[i])
        if cnt % 10 == 0:
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
    discriminator.load_state_dict(torch.load("Tmodel.tar"))
    discriminator.eval()
    preds=[]
    labels=[]
    senLabels=[]
    for words, pos, loc, maskL, maskR, label, senLabel, _ in dataset.batchs_test():
        words, pos, loc, maskL, maskR, label = words.to(device), pos.to(device), loc.to(device), maskL.to(device), maskR.to(device), label.to(device)
        loss, scores, pred = discriminator(words, pos, loc, maskL, maskR, label)
        preds.append(pred[1].cpu().numpy())
        labels.append(label.cpu().numpy())
        senLabels.extend(senLabel)
    preds=np.concatenate(preds,0)
    labels=np.concatenate(labels,0)
    senType={}
    for i in range(len(senLabels)):
        if preds[i]:
            senType[senLabels[i]]=preds[i]
    senLabels=[]
    for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel in dataset.batchs_gibbs():
        senLabels.extend(senLabel.cpu().numpy())
    subtype_pred=[]
    for x in senLabels:
        if x in senType:
            subtype_pred.append(senType[x])
        else:
            subtype_pred.append(0)
    np.save(dataPath + Tag + "_subtype_arg_pred.npy", np.array(subtype_pred))
    f1 = f1_score(labels, preds, labels = list(range(1, 34)), average = "micro")
    return f1

if __name__=='__main__':
    train()