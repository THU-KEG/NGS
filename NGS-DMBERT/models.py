from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *
from dataset import *
from pytorch_pretrained_bert.modeling import BertModel
import sys

def Print_file(file_name, item):
    with open('results/' + file_name, 'w') as f:
        f.write(file_name + ':\n')
        f.write(str(item.shape))
        for i in item:
            f.write(str(i) + '\n')

# Stage1. trigger classification
class DMCNN_Encoder(BertModel):
    def __init__(self, config):
        super(DMCNN_Encoder,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(p=keepProb)
        self.maxpooling=nn.MaxPool1d(SenLen)
    def forward(self, inp, inMask, maskL, maskR):
        SZ = inp.size(0)
        conved, _ = self.bert(inp, None, inMask, False)
        conved = conved.transpose(1,2)
        conved = conved.transpose(0,1)
        L = (conved * maskL).transpose(0,1)
        R = (conved * maskR).transpose(0,1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(SZ, dimC)
        pooledR = self.maxpooling(R).contiguous().view(SZ, dimC)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        rep=F.tanh(self.dropout(pooled))
        #rep=torch.tanh(pooled)
        return rep

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = DMCNN_Encoder.from_pretrained("../../BERT_CACHE/bert-base-uncased")
        self.M = nn.Linear(EncodedDim,dimE)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, inMask, maskL, maskR, label):
        reps = self.encoder(inp, inMask, maskL, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

# Stage2. argument classification
class DMCNN_Encoder_argument0(BertModel):
    def __init__(self,config):
        super(DMCNN_Encoder_argument0,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(p=keepProb)
        self.maxpooling=nn.MaxPool1d(SenLen)
    def forward(self, inp, inMask, maskL, maskM, maskR):
        SZ=inp.size(0)
        conved,_=self.bert(inp,None,inMask,False)
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        M=(conved*maskM).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        M=M+torch.ones_like(M)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(SZ,dimC)
        pooledM=self.maxpooling(M).contiguous().view(SZ,dimC)
        pooledR=self.maxpooling(R).contiguous().view(SZ,dimC)
        pooled=torch.cat((pooledL,pooledM,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
        rep=F.tanh(self.dropout(pooled))
        return rep

class Discriminator_argument0(nn.Module):
    def __init__(self):
        super(Discriminator_argument0, self).__init__()
        self.encoder = DMCNN_Encoder_argument0.from_pretrained("../../BERT_CACHE/bert-base-uncased")
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, inMask, maskL, maskM, maskR, label):
        reps = self.encoder(inp, inMask, maskL, maskM, maskR)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds


# Stage3. conditional probability
class DMCNN_Encoder_argument(BertModel):
    def __init__(self,config):
        super(DMCNN_Encoder_argument,self).__init__(config)
        self.bert=BertModel(config)
        self.dropout=nn.Dropout(p=keepProb)
        self.maxpooling=nn.MaxPool1d(SenLen)
    def forward(self, inp, inMask, maskL, maskM, maskR, argRole):
        SZ=inp.size(0)
        conved,_=self.bert(inp,argRole,inMask,False)
        conved=conved.transpose(1,2)
        conved=conved.transpose(0,1)
        L=(conved*maskL).transpose(0,1)
        M=(conved*maskM).transpose(0,1)
        R=(conved*maskR).transpose(0,1)
        L=L+torch.ones_like(L)
        M=M+torch.ones_like(M)
        R=R+torch.ones_like(R)
        pooledL=self.maxpooling(L).contiguous().view(SZ,dimC)
        pooledM=self.maxpooling(M).contiguous().view(SZ,dimC)
        pooledR=self.maxpooling(R).contiguous().view(SZ,dimC)
        pooled=torch.cat((pooledL,pooledM,pooledR),1)
        pooled=pooled-torch.ones_like(pooled)
        rep=F.tanh(self.dropout(pooled))
        return rep

class Discriminator_argument(nn.Module):
    def __init__(self):
        super(Discriminator_argument, self).__init__()
        self.encoder = DMCNN_Encoder_argument.from_pretrained("../../BERT_CACHE/bert-base-uncased_more")
        self.M = nn.Linear(EncodedDim_arg,dimR)
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inp, inMask, maskL, maskM, maskR, label, argRole):
        reps = self.encoder(inp, inMask, maskL, maskM, maskR, argRole)
        logits = self.M(reps)
        scores = torch.softmax(logits, dim = 1)
        loss = self.loss(logits, label)
        preds = torch.max(scores, dim = 1)
        return loss, scores, preds

