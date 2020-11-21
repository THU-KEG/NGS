import os
import re
import sys
import numpy as np
from constant import *
import torch
wordVec = np.load(wordvecPath + "wordVec.npy")
maskTag = 37

class Dataset:
    def __init__(self, Tag):
        print(Tag)
        self.tag = Tag
        self.words = np.load(dataPath + Tag + "_wordEmb.npy")
        self.pos = np.load(dataPath + Tag + "_posEmb.npy")
        self.loc = np.load(dataPath + Tag + "_local.npy")
        self.label = np.load(dataPath + Tag + "_label.npy")
        self.maskL = np.load(dataPath + Tag + "_maskL.npy")
        self.maskR = np.load(dataPath + Tag + "_maskR.npy")
        self.filename = []
        self.senLabel = np.load(dataPath + Tag + "_senLabel.npy")
        if Tag == 'test':
            with open(dataPath + Tag + "_filename", "r") as fp:
                filenames = fp.read().split()
                for filename in filenames:
                    self.filename.append(filename)

        self.words_arg = np.load(dataPath + Tag + "_wordEmb_arg.npy")
        self.pos1_arg = np.load(dataPath + Tag + "_pos1Emb_arg.npy")
        self.pos2_arg = np.load(dataPath + Tag + "_pos2Emb_arg.npy")
        self.loc_arg = np.load(dataPath + Tag + "_local_arg.npy")
        self.loc_mark_arg = np.load(dataPath + Tag + "_localMark_arg.npy")
        self.subtype_arg = np.load(dataPath + Tag + "_subtype_arg.npy")
        if os.path.exists(dataPath + Tag + "_subtype_arg_pred.npy"):
            self.subtype_arg_pred = np.load(dataPath + Tag + "_subtype_arg_pred.npy")
        if os.path.exists(dataPath + Tag + "_argRole_arg_noise.npy"):
            self.argRole_arg_noise = np.load(dataPath + Tag + "_argRole_arg_noise.npy")
        self.argRole_arg = np.load(dataPath + Tag + "_argRole_arg.npy")
        self.label_arg = np.load(dataPath + Tag + "_label_arg.npy")
        self.maskL_arg = np.load(dataPath + Tag + "_maskL_arg.npy")
        self.maskM_arg = np.load(dataPath + Tag + "_maskM_arg.npy")
        self.maskR_arg = np.load(dataPath + Tag + "_maskR_arg.npy")
        self.senLabel_arg = np.load(dataPath + Tag + "_Senlabel_arg.npy")
        self.ettIdx_arg = np.load(dataPath + Tag + "_ettIdx_arg.npy")
        self.ettLength_arg = np.load(dataPath + Tag + "_ettLength_arg.npy")

    def batchs(self):
        indices = np.random.permutation(np.arange(len(self.words)))
        self.words = self.words[indices]
        self.pos = self.pos[indices]
        self.loc = self.loc[indices]
        self.label = self.label[indices]
        self.maskL = self.maskL[indices]
        self.maskR = self.maskR[indices]
        for i in range(0, len(self.words)//BatchSize+1):
            L = i * BatchSize
            if L >= len(self.words):
                break
            R = min((i+1)*BatchSize, len(self.words))
            yield torch.LongTensor(self.words[L:R]), torch.LongTensor(self.pos[L:R]),\
                  torch.LongTensor(self.loc[L:R]), torch.FloatTensor(self.maskL[L:R]),\
                  torch.FloatTensor(self.maskR[L:R]), torch.LongTensor(self.label[L:R])

    def batchs_test(self):
        for i in range(0, len(self.words)//BatchSize+1):
            L = i * BatchSize
            if L >= len(self.words):
                break
            R = min((i+1)*BatchSize, len(self.words))
            yield torch.LongTensor(self.words[L:R]), torch.LongTensor(self.pos[L:R]),\
                  torch.LongTensor(self.loc[L:R]), torch.FloatTensor(self.maskL[L:R]),\
                  torch.FloatTensor(self.maskR[L:R]), torch.LongTensor(self.label[L:R]),\
                  self.senLabel[L:R], self.filename[L:R]

    def batchs_arg(self):
        # For argument stage
        indices = np.random.permutation(np.arange(len(self.words_arg)))
        self.words_arg = self.words_arg[indices]
        self.pos1_arg = self.pos1_arg[indices]
        self.pos2_arg = self.pos2_arg[indices]
        self.loc_arg = self.loc_arg[indices]
        self.loc_mark_arg = self.loc_mark_arg[indices]
        self.subtype_arg = self.subtype_arg[indices]
        self.subtype_arg_pred = self.subtype_arg_pred[indices]
        self.label_arg = self.label_arg[indices]
        self.maskL_arg = self.maskL_arg[indices]
        self.maskR_arg = self.maskR_arg[indices]
        self.maskM_arg = self.maskM_arg[indices]
        for i in range(0, len(self.words_arg) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.words_arg):
                break
            R = min((i+1)*BatchSize_arg, len(self.words_arg))
            yield torch.LongTensor(self.words_arg[L:R]), torch.LongTensor(self.pos1_arg[L:R]),\
                  torch.LongTensor(self.pos2_arg[L:R]), torch.LongTensor(self.loc_arg[L:R]),\
                  torch.LongTensor(self.loc_mark_arg[L:R]), torch.LongTensor(self.subtype_arg_pred[L:R]),\
                  torch.FloatTensor(self.maskL_arg[L:R]),\
                  torch.FloatTensor(self.maskM_arg[L:R]),torch.FloatTensor(self.maskR_arg[L:R]),\
                  torch.LongTensor(self.label_arg[L:R]), torch.LongTensor(self.subtype_arg[L:R])

    def batchs_argProb(self):
        # For conditional probability stage
        indices = np.random.permutation(np.arange(len(self.words_arg)))
        self.words_arg = self.words_arg[indices]
        self.pos1_arg = self.pos1_arg[indices]
        self.pos2_arg = self.pos2_arg[indices]
        self.loc_arg = self.loc_arg[indices]
        self.loc_mark_arg = self.loc_mark_arg[indices]
        self.subtype_arg = self.subtype_arg[indices]
        self.subtype_arg_pred = self.subtype_arg_pred[indices]
        self.argRole_arg = self.argRole_arg[indices]
        self.label_arg = self.label_arg[indices]
        self.maskL_arg = self.maskL_arg[indices]
        self.maskR_arg = self.maskR_arg[indices]
        self.maskM_arg = self.maskM_arg[indices]
        for i in range(0, len(self.words_arg) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.words_arg):
                break
            R = min((i+1)*BatchSize_arg, len(self.words_arg))
            yield torch.LongTensor(self.words_arg[L:R]), torch.LongTensor(self.pos1_arg[L:R]),\
                  torch.LongTensor(self.pos2_arg[L:R]), torch.LongTensor(self.loc_arg[L:R]),\
                  torch.LongTensor(self.loc_mark_arg[L:R]), torch.LongTensor(self.subtype_arg_pred[L:R]),\
                  torch.LongTensor(self.argRole_arg[L:R]), torch.FloatTensor(self.maskL_arg[L:R]),\
                  torch.FloatTensor(self.maskM_arg[L:R]), torch.FloatTensor(self.maskR_arg[L:R]),\
                  torch.LongTensor(self.label_arg[L:R]), torch.LongTensor(self.subtype_arg[L:R])

    def batchs_argProb_noise2(self):
        indices = np.random.permutation(np.arange(len(self.words_arg)))
        self.words_arg = self.words_arg[indices]
        self.pos1_arg = self.pos1_arg[indices]
        self.pos2_arg = self.pos2_arg[indices]
        self.loc_arg = self.loc_arg[indices]
        self.loc_mark_arg = self.loc_mark_arg[indices]
        self.subtype_arg = self.subtype_arg[indices]
        self.subtype_arg_pred = self.subtype_arg_pred[indices]
        self.argRole_arg_noise = self.argRole_arg_noise[indices]
        self.label_arg = self.label_arg[indices]
        self.maskL_arg = self.maskL_arg[indices]
        self.maskR_arg = self.maskR_arg[indices]
        self.maskM_arg = self.maskM_arg[indices]
        for i in range(0, len(self.words_arg) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.words_arg):
                break
            R = min((i+1)*BatchSize_arg, len(self.words_arg))
            yield torch.LongTensor(self.words_arg[L:R]), torch.LongTensor(self.pos1_arg[L:R]),\
                  torch.LongTensor(self.pos2_arg[L:R]), torch.LongTensor(self.loc_arg[L:R]),\
                  torch.LongTensor(self.loc_mark_arg[L:R]), torch.LongTensor(self.subtype_arg_pred[L:R]),\
                  torch.LongTensor(self.argRole_arg_noise[L:R]), torch.FloatTensor(self.maskL_arg[L:R]),\
                  torch.FloatTensor(self.maskM_arg[L:R]),torch.FloatTensor(self.maskR_arg[L:R]),\
                  torch.LongTensor(self.label_arg[L:R]), torch.LongTensor(self.subtype_arg[L:R])

    def batchs_gibbs(self):
        # For Gibbs Sampling
        for i in range(0, len(self.words_arg) // BatchSize_arg + 1):
            L = i * BatchSize_arg
            if L >= len(self.words_arg):
                break
            R = min((i+1)*BatchSize_arg, len(self.words_arg))
            if hasattr(self, "subtype_arg_pred"):
                yield torch.LongTensor(self.words_arg[L:R]), torch.LongTensor(self.pos1_arg[L:R]),\
                    torch.LongTensor(self.pos2_arg[L:R]), torch.LongTensor(self.loc_arg[L:R]),\
                    torch.LongTensor(self.loc_mark_arg[L:R]), torch.LongTensor(self.subtype_arg_pred[L:R]),\
                    torch.FloatTensor(self.maskL_arg[L:R]), torch.FloatTensor(self.maskM_arg[L:R]),\
                    torch.FloatTensor(self.maskR_arg[L:R]), torch.LongTensor(self.label_arg[L:R]),\
                    torch.LongTensor(self.ettIdx_arg[L:R]), torch.LongTensor(self.ettLength_arg[L:R]),\
                    torch.LongTensor(self.senLabel_arg[L:R]), torch.LongTensor(self.subtype_arg[L:R])
            else:
                yield torch.LongTensor(self.words_arg[L:R]), torch.LongTensor(self.pos1_arg[L:R]),\
                    torch.LongTensor(self.pos2_arg[L:R]), torch.LongTensor(self.loc_arg[L:R]),\
                    torch.LongTensor(self.loc_mark_arg[L:R]), torch.LongTensor(self.subtype_arg[L:R]),\
                    torch.FloatTensor(self.maskL_arg[L:R]), torch.FloatTensor(self.maskM_arg[L:R]),\
                    torch.FloatTensor(self.maskR_arg[L:R]), torch.LongTensor(self.label_arg[L:R]),\
                    torch.LongTensor(self.ettIdx_arg[L:R]), torch.LongTensor(self.ettLength_arg[L:R]),\
                    torch.LongTensor(self.senLabel_arg[L:R])