import pandas as pd
import pickle
import os
import numpy as np

class process:
    def __init__(self,parse):
        self.vocab = self.get_vocab(parse)
        self.data=self.vocab2int(parse)
        self.size=None


    def vocab2int(self,parse):
        data=[]
        if os.path.exists(parse.vocab_path+"pub.pkl"):
            data=pickle.load(open(parse.train_path+"pub.pkl","rb"))
            self.size=len(data)
        else:
            path=parse.vocab_path+parse.train_filename
            df=pd.read_csv(path)
            self.size=df.shape[0]
            for i in range(self.size):
                src=df.iloc[i]["src"]
                tgt=df.iloc[i]["tgt"]
                data.append([self.line2int(src,"src"),self.line2int(tgt,"tgt")])
            pickle.dump(data,open(parse.vocab_path+"pub.pkl","wb"))

        return data


    def line2int(self,line,st):
        li=[]
        for word in line:
            if word in self.vocab[st]:
                li.append(self.vocab[st][word])
            else :
                li.append(self.vocab[st]["<unk>"])
        return li

    def get_vocab(self,parse):
        vocab={"src":{},"tgt":{}}
        if os.path.exists(parse.vocab_path+parse.vocab_file):
            vocab=pickle.load(open(parse.vocab_path+parse.vocab_file,"rb"))
        else:
            if os.path.exists(parse.vocab_path+parse.src_vocab):
                with open(parse.vocab_path+parse.src_vocab,"r") as f:
                    count=0
                    for word in f:
                        if word not in vocab:
                            vocab["src"][word]=count
                            count+=1
            else:
                exit("You should set the resource file correct.")
            if os.path.exists(parse.vocab_path+parse.tgt_vocab):
                with open(parse.vocab_path+parse.tgt_vocab,"r") as f:
                    count=0
                    for word in f:
                        if word not in vocab["tgt"]:
                            vocab["tgt"][word]=count
                            count+=1
            else:
                exit("set the target vocab correct.")
            pickle.dump(vocab,open(parse.vocab_path+parse.vocab_file,"wb"))
        return vocab