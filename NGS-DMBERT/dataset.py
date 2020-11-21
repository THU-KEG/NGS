import os
import re
import sys
import numpy as np
from constant import *
import torch
maskTag = 37

class Dataset:
    def __init__(self, Tag):
        print(Tag)
        self.tag = Tag
        self.Twords = np.load(dataPath + Tag + "_wordEmb.npy")
        self.TinMask = np.load(dataPath + Tag + "_inMask.npy")
        self.Tlabel = np.load(dataPath + Tag + "_label.npy")
        self.TmaskL = np.load(dataPath + Tag + "_maskL.npy")
        self.TmaskR = np.load(dataPath + Tag + "_maskR.npy")
        self.Tfilename = []
        if Tag == 'test':
            self.TsenLabel = np.load(dataPath + Tag + "_senLabel.npy")

        self.Awords=np.load(dataPath + Tag + "_wordEmb_arg.npy")
        self.AinMask=np.load(dataPath + Tag + "_inMask_arg.npy")
        self.Alabel=np.load(dataPath + Tag + "_label_arg.npy")
        self.AmaskL=np.load(dataPath + Tag + "_maskL_arg.npy")
        self.AmaskM=np.load(dataPath + Tag + "_maskM_arg.npy")
        self.AmaskR=np.load(dataPath + Tag + "_maskR_arg.npy")
        self.Asubtype=np.load(dataPath + Tag + "_subtype_arg.npy")
        if os.path.exists(dataPath + Tag + "_subtype_arg_pred.npy"):
            self.Asubtype_pred = np.load(dataPath + Tag + "_subtype_arg_pred.npy")
        self.Aett_idx=np.load(dataPath + Tag + "_ettIdx_arg.npy")
        self.Aett_length=np.load(dataPath + Tag + "_ettLength_arg.npy")
        self.Atri_idx=np.load(dataPath + Tag + "_triIdx_arg.npy")
        self.Atri_length=np.load(dataPath + Tag + "_triLength_arg.npy")
        self.AargRole=np.load(dataPath + Tag + "_argRole_arg.npy")
        self.AsenLabel=np.load(dataPath + Tag + "_senLabel_arg.npy")

        self.TtrainNum=len(self.Twords)//BatchSize+1
        self.AtrainNum=len(self.Awords)//BatchSize_arg+1
        print (len(self.Awords), len(self.AargRole))

    def batchs(self):
        indices = np.random.permutation(np.arange(len(self.Twords)))
        self.Twords = self.Twords[indices]
        self.TinMask = self.TinMask[indices]
        self.Tlabel = self.Tlabel[indices]
        self.TmaskL = self.TmaskL[indices]
        self.TmaskR = self.TmaskR[indices]
        print (len(self.Twords), len(self.TinMask), len(self.Tlabel), len(self.TmaskL), len(self.TmaskR))

        for i in range(0, len(self.Twords)//BatchSize+1):
            L = i * BatchSize
            if L >= len(self.Twords):
                break
            R = min((i+1)*BatchSize, len(self.Twords))
            yield torch.LongTensor(self.Twords[L:R]), torch.LongTensor(self.TinMask[L:R]), \
                  torch.FloatTensor(self.TmaskL[L:R]), torch.FloatTensor(self.TmaskR[L:R]), \
                  torch.LongTensor(self.Tlabel[L:R])

    def batchs_test(self):
        for i in range(0, len(self.Twords)//BatchSize+1):
            L = i * BatchSize
            if L >= len(self.Twords):
                break
            R = min((i+1)*BatchSize, len(self.Twords))
            yield torch.LongTensor(self.Twords[L:R]), torch.LongTensor(self.TinMask[L:R]), \
                  torch.FloatTensor(self.TmaskL[L:R]), torch.FloatTensor(self.TmaskR[L:R]), \
                  torch.LongTensor(self.Tlabel[L:R]), self.TsenLabel[L:R]

    def batchs_test_arg(self):
        for i in range(0, len(self.Twords)//BatchSize+1):
            L = i * BatchSize_arg
            if L >= len(self.Twords):
                break
            R = min((i+1)*BatchSize, len(self.Twords))
            yield torch.LongTensor(self.Awords[L:R]), torch.LongTensor(self.AinMask[L:R]), \
                  torch.FloatTensor(self.AmaskL[L:R]), torch.FloatTensor(self.AmaskM[L:R]), \
                  torch.FloatTensor(self.AmaskR[L:R]), torch.LongTensor(self.Alabel[L:R]), self.AsenLabel[L:R], \
                  torch.LongTensor(self.Aett_idx[L:R]), torch.LongTensor(self.Aett_length[L:R]), \
                  torch.LongTensor(self.Atri_idx[L:R]), torch.LongTensor(self.AargRole[L:R]), self.Asubtype[L:R], self.Asubtype_pred[L:R]

    def batchs_arg(self):
        indices = np.random.permutation(np.arange(len(self.Awords)))
        self.Awords = self.Awords[indices]
        self.AinMask = self.AinMask[indices]
        self.Alabel = self.Alabel[indices]
        self.AmaskL = self.AmaskL[indices]
        self.AmaskR = self.AmaskR[indices]
        self.AmaskM = self.AmaskM[indices]
        self.Asubtype = self.Asubtype[indices]
        self.Asubtype_pred = self.Asubtype_pred[indices]

        for i in range(0, len(self.Awords) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.Awords):
                break
            R = min((i+1)*BatchSize_arg, len(self.Awords))
            yield torch.LongTensor(self.Awords[L:R]), torch.LongTensor(self.AinMask[L:R]), \
                  torch.FloatTensor(self.AmaskL[L:R]), torch.FloatTensor(self.AmaskM[L:R]), \
                  torch.FloatTensor(self.AmaskR[L:R]), torch.LongTensor(self.Alabel[L:R]), self.Asubtype[L:R], self.Asubtype_pred[L:R]


    def batchs_argProb(self):
        indices = np.random.permutation(np.arange(len(self.Awords)))
        self.Awords = self.Awords[indices]
        self.AinMask = self.AinMask[indices]
        self.Alabel = self.Alabel[indices]
        self.AmaskL = self.AmaskL[indices]
        self.AmaskR = self.AmaskR[indices]
        self.AmaskM = self.AmaskM[indices]
        self.AargRole = self.AargRole[indices]
        self.Asubtype = self.Asubtype[indices]
        self.Asubtype_pred = self.Asubtype_pred[indices]

        for i in range(0, len(self.Awords) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.Awords):
                break
            R = min((i+1)*BatchSize_arg, len(self.Awords))
            yield torch.LongTensor(self.Awords[L:R]), torch.LongTensor(self.AinMask[L:R]), \
                  torch.FloatTensor(self.AmaskL[L:R]), torch.FloatTensor(self.AmaskM[L:R]), \
                  torch.FloatTensor(self.AmaskR[L:R]), torch.LongTensor(self.AargRole[L:R]), \
                  torch.LongTensor(self.Alabel[L:R]), self.Asubtype[L:R], self.Asubtype_pred[L:R]

    def load_argProb_noise(self):
        self.AargRole=np.load(dataPath + self.tag + "_argRole_arg_noise.npy")

    def batchs_gibbs(self):
        for i in range(0, len(self.Awords) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.Awords):
                break
            R = min((i+1)*BatchSize_arg, len(self.Awords))
            if hasattr(self, "Atype_pred"):
                yield torch.LongTensor(self.Awords[L:R]), torch.LongTensor(self.AinMask[L:R]), \
                    torch.FloatTensor(self.AmaskL[L:R]), torch.FloatTensor(self.AmaskM[L:R]), \
                    torch.FloatTensor(self.AmaskR[L:R]), torch.LongTensor(self.Alabel[L:R]), \
                    self.Aett_idx[L:R], self.Aett_length[L:R], \
                    torch.LongTensor(self.AargRole[L:R]), self.AsenLabel[L:R], self.Asubtype[L:R], self.Asubtype_pred[L:R]
            else:
                yield torch.LongTensor(self.Awords[L:R]), torch.LongTensor(self.AinMask[L:R]), \
                    torch.FloatTensor(self.AmaskL[L:R]), torch.FloatTensor(self.AmaskM[L:R]), \
                    torch.FloatTensor(self.AmaskR[L:R]), torch.LongTensor(self.Alabel[L:R]), \
                    self.Aett_idx[L:R], self.Aett_length[L:R], \
                    torch.LongTensor(self.AargRole[L:R]), self.AsenLabel[L:R]