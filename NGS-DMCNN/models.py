from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *
from dataset import *
import sys

def Print_file(file_name, item):
    with open('results/' + file_name, 'w') as f:
        f.write(file_name + ':\n')
        f.write(str(item.shape))
        for i in item:
            f.write(str(i) + '\n')

# Stage1. trigger classification
class DMCNN_Encoder(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder,self).__init__()
        self.word_emb = nn.Embedding(len(wordVec), dimWE, padding_idx = 0)
        weight = torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        self.pos_emb = nn.Embedding(MaxPos, dimPE)
        self.conv = nn.Conv1d(dimWE + dimPE, dimC, filter_size, padding = 1)
        self.dropout = nn.Dropout(p = keepProb)
        self.maxpooling = nn.MaxPool1d(SenLen)
    def forward(self, inp, pos, loc, maskL, maskR):
        SZ = inp.size(0)
        embeds = self.word_emb(inp)
        pos_embeds = self.pos_emb(pos)
        loc_embeds = self.word_emb(loc).contiguous().view(SZ, (2*LocalLen+1)*dimWE)
        wordVec = torch.cat((embeds,pos_embeds),2).transpose(1,2)
        conved = self.conv(wordVec)
        conved = conved.transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ, dimC)
        pooledR = self.maxpooling(R).contiguous().view(SZ, dimC)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        rep = torch.cat((pooled, loc_embeds), 1)
        rep = torch.tanh(self.dropout(rep))
        return rep

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = DMCNN_Encoder()
        self.M = nn.Linear(EncodedDim,dimE)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, pos, loc, maskL, maskR, label):
        reps = self.encoder(inp, pos, loc, maskL, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

# Stage2. argument classification
class DMCNN_Encoder_argument(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder_argument,self).__init__()
        self.word_emb = nn.Embedding(len(wordVec), dimWE, padding_idx = 0)
        #self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(wordVec), freeze = False, padding_idx = 0)
        weight = torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        #self.word_emb.weight.requires_grad_(True)
        #print(self.word_emb.weight.data[0])
        self.pos_emb = nn.Embedding(MaxPos, dimPE)
        self.event_emb = nn.Embedding(dimR, dimEE, padding_idx = 0)
        self.conv = nn.Conv1d(dimWE + 2 * dimPE + dimEE, dimC_arg, filter_size, padding = 1)
        self.dropout = nn.Dropout(p = keepProb)
        #self.M = nn.Linear(EncodedDim,dimE)
        self.maxpooling = nn.MaxPool1d(SenLen)
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR):
        SZ = inp.size(0)
        embeds = self.word_emb(inp)
        pos1_embeds = self.pos_emb(pos1)
        pos2_embeds = self.pos_emb(pos2)
        loc_temp = self.word_emb(loc).contiguous().view(SZ, -1, dimWE)
        loc_embeds = []
        for idx, l in enumerate(loc_temp):
            l[4] = torch.mean(l[4: 4 + loc_mark[idx]], 0, True)
            l[5] = l[4 + loc_mark[idx]]
            l = l[0:6].view(-1)
            loc_embeds.append(l)
        loc_embeds = torch.stack(loc_embeds)
        event_embeds = self.event_emb(subtype).repeat(1, SenLen).view(SZ, SenLen, dimEE)
        wordVec = torch.cat((embeds, pos1_embeds, pos2_embeds, event_embeds), 2).transpose(1,2)
        conved = self.conv(wordVec).transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        M = (conved * maskM).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        M = M + torch.ones_like(M)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ,dimC_arg)
        pooledM = self.maxpooling(M).contiguous().view(SZ,dimC_arg)
        pooledR = self.maxpooling(R).contiguous().view(SZ,dimC_arg)
        pooled = torch.cat((pooledL,pooledM,pooledR),1)
        pooled = pooled - torch.ones_like(pooled)
        rep = torch.cat((pooled,loc_embeds),1)
        rep = torch.tanh(self.dropout(rep))
        return rep

class Discriminator_argument(nn.Module):
    def __init__(self):
        super(Discriminator_argument, self).__init__()
        self.encoder = DMCNN_Encoder_argument()
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR, label):
        reps = self.encoder(inp, pos1, pos2, loc, loc_mark, subtype, maskL, maskM, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

# Stage3(1). conditional probability VERSION1
class DMCNN_Encoder_argProb(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder_argProb,self).__init__()
        self.word_emb = nn.Embedding(len(wordVec), dimWE, padding_idx = 0)
        weight = torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        self.pos_emb = nn.Embedding(MaxPos, dimPE)
        self.event_emb = nn.Embedding(dimR, dimEE, padding_idx = 0)
        self.role_emb = nn.Embedding(dimR, dimRE, padding_idx = 0)
        self.conv = nn.Conv1d(dimWE + 2 * dimPE + dimEE + (maxE-1) * dimRE, dimC_arg, filter_size, padding = 1)
        self.dropout = nn.Dropout(p = keepProb)
        self.maxpooling = nn.MaxPool1d(SenLen)
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR):
        SZ = inp.size(0)
        embeds = self.word_emb(inp)
        pos1_embeds = self.pos_emb(pos1)
        pos2_embeds = self.pos_emb(pos2)
        loc_temp = self.word_emb(loc).contiguous().view(SZ, -1, dimWE)
        loc_embeds = []
        for idx, l in enumerate(loc_temp):
            l[4] = torch.mean(l[4: 4 + loc_mark[idx]], 0, True)
            l[5] = l[4 + loc_mark[idx]]
            l = l[0:6].view(-1)
            loc_embeds.append(l)
        loc_embeds = torch.stack(loc_embeds)
        event_embeds = self.event_emb(subtype).repeat(1, SenLen).view(SZ, SenLen, dimEE)
        role_embeds = self.role_emb(argRole).repeat(1, SenLen, 1).view(SZ, SenLen, (maxE-1) * dimRE)
        wordVec = torch.cat((embeds, pos1_embeds, pos2_embeds, event_embeds, role_embeds), 2).transpose(1,2)
        conved = self.conv(wordVec).transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        M = (conved * maskM).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        M = M + torch.ones_like(M)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ,dimC_arg)
        pooledM = self.maxpooling(M).contiguous().view(SZ,dimC_arg)
        pooledR = self.maxpooling(R).contiguous().view(SZ,dimC_arg)
        pooled = torch.cat((pooledL,pooledM,pooledR),1)
        pooled = pooled - torch.ones_like(pooled)
        rep = torch.cat((pooled,loc_embeds),1)
        rep = torch.tanh(self.dropout(rep))
        return rep

class Discriminator_argProb(nn.Module):
    def __init__(self):
        super(Discriminator_argProb, self).__init__()
        self.encoder = DMCNN_Encoder_argProb()
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR, label):
        reps = self.encoder(inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

# Stage3(2). conditional probability VERSION2
class DMCNN_Encoder_argProb2(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder_argProb2,self).__init__()
        self.word_emb = nn.Embedding(len(wordVec), dimWE, padding_idx = 0)
        #self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(wordVec), freeze = False, padding_idx = 0)
        weight = torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        self.pos_emb = nn.Embedding(MaxPos, dimPE)
        self.event_emb = nn.Embedding(dimR, dimEE, padding_idx = 0)
        self.role_emb = nn.Embedding(dimR + 2, dimRE, padding_idx = 0)
        self.conv = nn.Conv1d(dimWE + 2 * dimPE + dimEE + dimRE, dimC_arg, filter_size, padding = 1)
        self.dropout = nn.Dropout(p = keepProb)
        self.maxpooling = nn.MaxPool1d(SenLen)
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR):
        SZ = inp.size(0)
        embeds = self.word_emb(inp)
        pos1_embeds = self.pos_emb(pos1)
        pos2_embeds = self.pos_emb(pos2)
        loc_temp = self.word_emb(loc).contiguous().view(SZ, -1, dimWE)
        loc_embeds = []
        for idx, l in enumerate(loc_temp):
            l[4] = torch.mean(l[4: 4 + loc_mark[idx]], 0, True)
            l[5] = l[4 + loc_mark[idx]]
            l = l[0:6].view(-1)
            loc_embeds.append(l)
        loc_embeds = torch.stack(loc_embeds)
        event_embeds = self.event_emb(subtype).repeat(1, SenLen).view(SZ, SenLen, dimEE)
        role_embeds = self.role_emb(argRole)
        wordVec = torch.cat((embeds, pos1_embeds, pos2_embeds, event_embeds, role_embeds), 2).transpose(1,2)
        conved = self.conv(wordVec).transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        M = (conved * maskM).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        M = M + torch.ones_like(M)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ,dimC_arg)
        pooledM = self.maxpooling(M).contiguous().view(SZ,dimC_arg)
        pooledR = self.maxpooling(R).contiguous().view(SZ,dimC_arg)
        pooled = torch.cat((pooledL,pooledM,pooledR),1)
        pooled = pooled - torch.ones_like(pooled)
        rep = torch.cat((pooled,loc_embeds),1)
        rep = torch.tanh(self.dropout(rep))
        return rep

class Discriminator_argProb2(nn.Module):
    def __init__(self):
        super(Discriminator_argProb2, self).__init__()
        self.encoder = DMCNN_Encoder_argProb2()
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR, label):
        reps = self.encoder(inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

# Stage3(3). conditional probability with noise VERSION2
class DMCNN_Encoder_argProb2_noise2(nn.Module):
    def __init__(self):
        super(DMCNN_Encoder_argProb2_noise2,self).__init__()
        self.word_emb = nn.Embedding(len(wordVec), dimWE, padding_idx = 0)
        #self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(wordVec), freeze = False, padding_idx = 0)
        weight = torch.tensor(wordVec)
        weight.requires_grad_(True)
        self.word_emb.weight.data.copy_(weight)
        self.pos_emb = nn.Embedding(MaxPos, dimPE)
        self.event_emb = nn.Embedding(dimR, dimEE, padding_idx = 0)
        self.role_emb = nn.Embedding(dimR + 2, dimRE_noise, padding_idx = 0)
        self.conv = nn.Conv1d(dimWE + 2 * dimPE + dimEE + dimRE_noise, dimC_arg, filter_size, padding = 1)
        self.dropout = nn.Dropout(p = keepProb)
        self.maxpooling = nn.MaxPool1d(SenLen)
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR):
        SZ = inp.size(0)
        embeds = self.word_emb(inp)
        pos1_embeds = self.pos_emb(pos1)
        pos2_embeds = self.pos_emb(pos2)
        loc_temp = self.word_emb(loc).contiguous().view(SZ, -1, dimWE)
        loc_embeds = []
        for idx, l in enumerate(loc_temp):
            l[4] = torch.mean(l[4: 4 + loc_mark[idx]], 0, True)
            l[5] = l[4 + loc_mark[idx]]
            l = l[0:6].view(-1)
            loc_embeds.append(l)
        loc_embeds = torch.stack(loc_embeds)
        event_embeds = self.event_emb(subtype).repeat(1, SenLen).view(SZ, SenLen, dimEE)
        role_embeds = self.role_emb(argRole)
        wordVec = torch.cat((embeds, pos1_embeds, pos2_embeds, event_embeds, role_embeds), 2).transpose(1,2)
        conved = self.conv(wordVec).transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        M = (conved * maskM).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        M = M + torch.ones_like(M)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ,dimC_arg)
        pooledM = self.maxpooling(M).contiguous().view(SZ,dimC_arg)
        pooledR = self.maxpooling(R).contiguous().view(SZ,dimC_arg)
        pooled = torch.cat((pooledL,pooledM,pooledR),1)
        pooled = pooled - torch.ones_like(pooled)
        rep = torch.cat((pooled,loc_embeds),1)
        rep = torch.tanh(self.dropout(rep))
        return rep

class Discriminator_argProb2_noise2(nn.Module):
    def __init__(self):
        super(Discriminator_argProb2_noise2, self).__init__()
        self.encoder = DMCNN_Encoder_argProb2_noise2()
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR, label):
        reps = self.encoder(inp, pos1, pos2, loc, loc_mark, subtype, argRole, maskL, maskM, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds
