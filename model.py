import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed):
        super(EncoderDecoder,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decode(self.encode(src),tgt,src_mask,tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,tgt,src_mask,tgt_mask):
        return self.decoder(tgt.tgt_embed(tgt),memory,src_mask,tgt_mask)

def clone(layer,N):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self,size,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_1=nn.Parameter(torch.ones(size))
        self.b_1=nn.Parameter(torch.zeros(size))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_1*(x-mean)/(std+self.eps)+self.b_1

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))


class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.list=clone(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        for l in self.list:
            x=l(x,mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.list=clone(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        for l in self.list:
            x=l(x,memory,src_mask,tgt_mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self,size,h,ff_model,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.atten=MultiHeaderAttention(h,size,dropout)
        self.feed_forward=PositionwiseFeedForward(size,ff_model,dropout)
        self.sublayer=clone(SublayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x: self.atten(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self,size,h,ff_model,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn=MultiHeaderAttention(h,size,dropout)
        self.src_atten=MultiHeaderAttention(h,size,dropout)
        self.feed_forward=PositionwiseFeedForward(size,ff_model,dropout)
        self.sublayers=clone(SublayerConnection(size,dropout),3)
        self.size=size
    def forward(self,x,memory,src_mask,tgt_mask):
        m=memory
        x=self.sublayers[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x=self.sublayers[1](x,lambda x: self.src_atten(x,m,m,src_mask))
        return self.sublayers[-1](x,self.feed_forward)

def attention(q,k,v,mask,dropout):
    d_k=q.size(-1)
    scores=torch.matmul(q,k.transpose(2,3))\
           /torch.sqrt(d_k)

    if mask is not None:
        scores=scores.masked_fill(mask,-1e9)

    p_attn=F.softmax(scores,dim=-1)
    if dropout is not None:
        p_attn=dropout(p_attn)

    return torch.matmul(p_attn,v),p_attn

class MultiHeaderAttention(nn.Module):
    def __init__(self,d_headers,d_model,dropout=0.1):
        super(MultiHeaderAttention,self).__init__()
        assert d_model%d_headers==0
        self.d_k=d_model//d_headers
        self.h=d_headers
        self.linears=clone(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
        nbatch=q.size(0)
        q,k,b=[l(x).view(nbatch,-1,self.h,self.d_k).\
                   transpose(1,2) for l,x in zip(self.linears,[q,k,v])]
        x,self.attn=attention(q,k,v,mask,dropout=self.dropout)
        x=x.transpose(1,2).contiguous()\
            .view(nbatch,-1,self.d_k*self.h)

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,ff_model,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,ff_model)
        self.w2=nn.Linear(ff_model,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w2(self.dropout(self.w1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2)*\
                           -(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

def build_model():
    pass