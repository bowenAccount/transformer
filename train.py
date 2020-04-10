import torch
import os
from torch.optim import Adam
from H_parse import H_parse
from model import build_model
from optimizer import *
from process_data import process




parse=H_parse()

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if parse.device=="cpu":
    DEVICE=torch.device("cpu")
data=process(parse)
model=build_model()
optim=NoamOpt(model.src_embed[0].d_model,1,4000,Adam(model.parameters(),lr=parse.lr_rate,betas=[0.9,0.98],eps=1e-9))
label_sm=LabelSmoothing(parse.tgt_vocab_len,data.vocab["tgt"]["<pad>"],smoothing=0.1)
losscompute=LossCompute(model.generator,label_sm,optim)