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
from pytorch_pretrained_bert import BertTokenizer

predEtt_tag = 36
Ett_tag = 36
tokenizer = BertTokenizer.from_pretrained("../BERT_CACHE/bert-base-uncased-vocab.txt")

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

# TrainSet = Dataset("train")
# DevelopSet = Dataset("develop")
TestSet = Dataset("test")
acc_NA = Accuracy()
acc_not_NA = Accuracy()
acc_total = Accuracy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get_prob2 = Discriminator_argument()
# get_prob2.to(device, non_blocking=True)
#get_prob2.load_state_dict(torch.load("Pmodel_bert_noise.tar"))

get_prob = Discriminator_argument()
get_prob.to(device, non_blocking=True)
get_prob.load_state_dict(torch.load("Pmodel_bert.tar"))

discriminator = Discriminator_argument()
discriminator.to(device, non_blocking=True)
discriminator.load_state_dict(torch.load("Amodel_bert.tar"))

if torch.cuda:
	torch.cuda.set_device(device)

Aword_prob = []
Ain_mask_prob = []
AmaskR_prob = []
AmaskL_prob = []
AmaskM_prob = []

def gen_Ains(word, tri_idx, ett_idx, ett_length, subtype_insert):
    word = word[:SenLen]
    if np.where(word == 0)[0] != []:
        length = np.where(word == 0)[0][0]
        word[length] = subtype_insert
        length += 1
    else:
        length = SenLen-1
    maskR = np.zeros((SenLen), dtype = np.float32)
    maskL = np.zeros((SenLen), dtype = np.float32)
    maskM = np.zeros((SenLen), dtype = np.float32)
    index_min = min(tri_idx, ett_idx + ett_length - 1)
    index_max = max(tri_idx, ett_idx + ett_length - 1)

    for j in range(SenLen):
        if j >= length:
            maskL[j] = 0
            maskM[j] = 0
            maskR[j] = 0
        elif j - index_min <= 0:
            maskL[j] = 1
            maskM[j] = 0
            maskR[j] = 0
        elif j - index_max <= 0:
            maskL[j] = 0
            maskM[j] = 1
            maskR[j] = 0
        else:
            maskL[j] = 0
            maskM[j] = 0
            maskR[j] = 1

    input_mask = [1]*length
    oriLen = length
    if length > SenLen:
        input_mask = input_mask[:SenLen]
    else:
        L = length
        for i in range(0, SenLen - L):
            input_mask.append(0)

    return word, input_mask, maskR, maskL, maskM

def random_index(rate):
    start = 0
    index = 0
    randnum = random.random()
    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return index

def gen_argRole(words, preds, ettIdxs, ettLengths, triIdxs, triLengths, subtypes):
    Aword_prob_temp = []
    Ain_mask_prob_temp = []
    AmaskR_prob_temp = []
    AmaskL_prob_temp = []
    AmaskM_prob_temp = []    
    for i in range(len(preds)):
        role_list = []
        idx_list = []
        word = words[i].cpu().numpy()
        tri_idx = triIdxs[i] - 1
        tri_length = triLengths[i] - 1
        ett_idx = ettIdxs[i]
        ett_length = ettLengths[i]
        for j in range(len(preds)):
            if ettIdxs[j] >= tri_idx:
                ettIdxs[j] -= 2
        subtype = subtypes[i] + 37
        index = np.where(word == subtype + 1)
        word = np.delete(word, index)
        word = np.append(word, 0)
        word = np.append(word, 0)

        for j in range(len(preds)):
            if i != j:
                role_list.append(preds[j])
                role_list.append(preds[j])
            else:
                role_list.append(0)
                role_list.append(0)
            idx_list.append(ettIdxs[j])
            idx_list.append(ettIdxs[j] + ettLengths[j])

        data = [(idx, role) for idx, role in zip(idx_list,role_list)]
        data.sort(reverse = True)
        idx_list = [idx for idx,role in data]
        role_list = [role for idx,role in data]

        for k, idx in enumerate(idx_list): 
            if idx < SenLen:
                if role_list[k] == 0:
                    insert_token = Ett_tag + 1
                else:
                    insert_token = role_list[k] + 1
                word = np.insert(word, idx, insert_token)

                if ett_idx >= idx:
                    ett_idx += 1
                elif ett_idx < idx and ett_idx + ett_length >= idx:
                    ett_length += 1
                if tri_idx >= idx:
                    tri_idx += 1
                elif tri_idx < idx and tri_idx + tri_length >= idx:
                    tri_length += 1

        word_prob, input_mask_prob, maskR_prob, maskL_prob, maskM_prob = gen_Ains(word, tri_idx + tri_length - 1, ett_idx, ett_length, subtype + 1)
        Aword_prob_temp.append(word_prob)
        Ain_mask_prob_temp.append(input_mask_prob)
        AmaskR_prob_temp.append(maskR_prob)
        AmaskL_prob_temp.append(maskL_prob)
        AmaskM_prob_temp.append(maskM_prob)
    return Aword_prob_temp, Ain_mask_prob_temp, AmaskR_prob_temp, AmaskM_prob_temp, AmaskL_prob_temp
                

def gibbs(dataset):
    discriminator.eval()
    get_prob.eval()
    # get_prob2.eval()
    preds = []
    labels = []
    subtypes = []
    subtypes_pred = []
    senLabels = []
    ettIdxs = []
    ettLengths = []
    triIdxs = []
    triLengths = []
    results = []
    words_ori = []

    # get the pred0
    for words, inMask, maskL, maskM, maskR, label, ettIdx, ettLength, triIdx, triLength, subtype, senLabel, subtype, subtype_pred in dataset.batchs_gibbs():
        words, inMask, maskL, maskM, maskR, label = words.to(device), inMask.to(device), maskL.to(device), maskM.to(device), maskR.to(device), label.to(device)       
        loss, scores, pred = discriminator(words, inMask, maskL, maskM, maskR, label)
        preds.extend(pred[1].cpu().numpy())
        labels.extend(label.cpu().numpy())
        words_ori.extend(words)
        senLabels.extend(senLabel)
        subtypes.extend(subtype)
        subtypes_pred.extend(subtype_pred)
        ettIdxs.extend(ettIdx)
        ettLengths.extend(ettLength)
        triIdxs.extend(triIdx)
        triLengths.extend(triLength)
        for i in range(len(words)):
            state = {}
            results.append(state)
    preds0 = copy.deepcopy(preds)

    # Transfer for N+K times

    for trans_time in range(int(1/k_an)):
        if trans_time % 5 == 0:
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
            acc, _, f1 = f_score(preds, labels, subtypes_pred, subtypes)
            print ("trans_time:{}, acc:{}, f1:{}".format(trans_time,acc,f1))

        # generate the argRoles
        L = 0
        R = 0
        sen_idx = 0
        Aword_prob = []
        Ain_mask_prob = []
        AmaskR_prob = []
        AmaskL_prob = []
        AmaskM_prob = []
    
        for i in range(len(senLabels)):
            if senLabels[i] == sen_idx:
                R += 1
            else:
                sen_idx = senLabels[i]
                if L != R:
                    Aword, Ain_mask, AmaskR, AmaskM, AmaskL = gen_argRole(words_ori[L:R], preds[L:R], ettIdxs[L:R], ettLengths[L:R], triIdxs[L:R], triLengths[L:R], subtypes[L:R])
                    Aword_prob.extend(Aword)
                    Ain_mask_prob.extend(Ain_mask)
                    AmaskR_prob.extend(AmaskR)
                    AmaskM_prob.extend(AmaskM)
                    AmaskL_prob.extend(AmaskL)
                    L = R
                    R += 1
            if i == len(senLabels) - 1:
                Aword, Ain_mask, AmaskR, AmaskM, AmaskL = gen_argRole(words_ori[L:R], preds[L:R], ettIdxs[L:R], ettLengths[L:R], triIdxs[L:R], triLengths[L:R], subtypes[L:R])
                Aword_prob.extend(Aword)
                Ain_mask_prob.extend(Ain_mask)
                AmaskR_prob.extend(AmaskR)
                AmaskM_prob.extend(AmaskM)
                AmaskL_prob.extend(AmaskL)

        # get prob
        probs = []
        probs_max = []
        probs_argmax = []
        words_sum = []
        Aword_prob = np.array(Aword_prob)
        Ain_mask_prob = np.array(Ain_mask_prob)
        AmaskR_prob = np.array(AmaskR_prob)
        AmaskM_prob = np.array(AmaskM_prob)
        AmaskL_prob = np.array(AmaskL_prob)

        L = 0
        for i in range(0, len(Aword_prob) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(Aword_prob):
                break
            R = min((i+1)*BatchSize_arg, len(Aword_prob))
            words, inMask, maskR, maskM, maskL, label = torch.LongTensor(Aword_prob[L:R]), torch.LongTensor(Ain_mask_prob[L:R]), torch.FloatTensor(AmaskR_prob[L:R]), torch.FloatTensor(AmaskM_prob[L:R]), torch.FloatTensor(AmaskL_prob[L:R]), torch.LongTensor(labels[L:R])
            words, inMask, maskR, maskM, maskL, label = words.to(device), inMask.to(device), maskR.to(device), maskM.to(device), maskL.to(device), label.to(device)
            _, prob, _ = get_prob(words, inMask, maskR, maskM, maskL, label)
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