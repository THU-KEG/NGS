from __future__ import print_function
from constant import *
from models import *
from dataset import *
from utils import f_score
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score,precision_score,recall_score
import random
import math
import json
import copy
from time import time

predEtt_tag = 36
Ett_tag = 36
argRoles = []

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

TestSet = Dataset("test")
acc_NA = Accuracy()
acc_not_NA = Accuracy()
acc_total = Accuracy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

get_prob = Discriminator_argProb2()
get_prob.to(device, non_blocking=True)
get_prob.load_state_dict(torch.load("Pmodel_noise.tar", map_location='cpu'))

discriminator = Discriminator_argument()
discriminator.to(device, non_blocking=True)
discriminator.load_state_dict(torch.load("Amodel.tar", map_location='cpu'))

def random_index(rate):
    start = 0
    index = 0
    randnum = random.random()
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def gen_argRole(pred, ettIdx, ettLength):
    global argRoles
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


def gibbs(dataset):
    discriminator.eval()
    get_prob.eval()
    # get_prob2.eval()
    preds = []
    labels = []
    pred_subtypes=[]
    golden_subtypes=[]
    senLabels = []
    ettIdxs = []
    ettLengths = []
    results = []

    # get the pred0
    for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel, subtype_golden in dataset.batchs_gibbs():
        words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel = words.to(device), pos1.to(device), pos2.to(device), loc.to(device), loc_mark.to(device), subtype.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device), ettIdx.to(device), ettLength.to(device), senLabel.to(device)
        loss, scores, pred = discriminator(words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label)
        preds.extend(pred[1].cpu().numpy())
        labels.extend(label.cpu().numpy())
        pred_subtypes.extend(subtype.cpu().numpy())
        golden_subtypes.extend(subtype_golden.cpu().numpy())
        senLabels.extend(senLabel.cpu().numpy().tolist())
        ettIdxs.extend(ettIdx)
        ettLengths.extend(ettLength)
        for i in range(len(words)):
            state = {}
            results.append(state)
    preds0 = copy.deepcopy(preds)

    # Transfer for N+K times
    global argRoles
    for trans_time in range(int(1/k_an) - 1):
        if trans_time % 10 == 0:
            cnt=0
            cnt1=0
            FN=0
            FP=0
            for i in range(len(preds)):
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
            print ("trans_time:{}, acc:{}, f1:{}".format(trans_time,acc,f1))

        # generate the argRoles
        L = 0
        R = 0
        sen_idx = senLabels[0]
        argRoles = []
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

        # get prob
        probs = []
        probs_max = []
        probs_argmax = []
        words_sum = []
        L = 0
        for words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel, subtype_golden in dataset.batchs_gibbs():
            argRole = torch.LongTensor(argRoles[L: L+len(words)]).to(device)
            L += len(words)
            words, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label, ettIdx, ettLength, senLabel = words.to(device), pos1.to(device), pos2.to(device), loc.to(device), loc_mark.to(device), subtype.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device), ettIdx.to(device), ettLength.to(device), senLabel.to(device)
            _, prob, _ = get_prob(words, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR, label)
            prob_max, prob_argmax = torch.max(prob, dim = 1)
            prob_max = prob_max.detach().cpu().numpy().tolist()
            prob_argmax = prob_argmax.detach().cpu().numpy().tolist()
            prob = prob.detach().cpu().numpy().tolist()

            words_sum.extend(words)
            probs.extend(prob)
            probs_max.extend(prob_max)
            probs_argmax.extend(prob_argmax)

        # transfer and sum the states
        probs_max_an = []
        sen_idx = 0
        L = 0
        R = 0
        for i, prob in enumerate(probs):
            if senLabels[i] == sen_idx:
                R += 1
            else: 
                sen_idx = senLabels[i]
                if L != R:
                    probMax_sum = 0
                    probs_max_an = []
                    for idx in range(L, R):
                        probMax_sum += probs_max[idx] ** (1/(1 - k_an * trans_time))
                    for idx in range(L, R):
                        probs_max_an.append(probs_max[idx] ** (1/(1 - k_an * trans_time)) / probMax_sum)
                    idx_trans = random_index(probs_max_an) + L
                    preds[idx_trans] = probs_argmax[idx_trans]
                    L = R
                    R += 1
        
    # print the results
    sen_idx = 0
    L = 0
    R = 0
    for i in range(len(senLabels)):
        if senLabels[i] == sen_idx:
            R += 1
        else:
            sen_idx = senLabels[i]
            if L != R:
                pred1 = np.zeros((R-L), dtype = np.int64)
                print ('---sentence%d---'%sen_idx)
                print ('pred0:', preds0[L:R])
                print ('label:', labels[L:R])
                print ('result:', preds[L:R])
                L = R
                R += 1
                
if __name__ == '__main__':
    gibbs(TestSet)