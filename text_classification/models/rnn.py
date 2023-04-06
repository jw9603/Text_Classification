import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    
    def __init__(self,input_size,word_vec_size,hidden_size,n_class,n_layers=4,dropout_p=.3):
        
        self.input_size = input_size # vocab size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super().__init__()
        
        self.emb = nn.Embedding(input_size,word_vec_size)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(input_size=word_vec_size,
                           hidden_size=hidden_size,
                           num_layers=n_layers,
                           dropout_p=dropout_p,
                           batch_first=True,
                           bidirectional=True)
        self.gen = nn.Linear(hidden_size*2, n_class) # because of bidirectional LSTM
        
        self.activation = nn.LogSoftmax(dim=-1)
        
    def forward(self,x):
        
        x = self.emb(x) # |x| = (batch, 1) or (batch,length)
        
        # |x| = (bs, length, ws)
        x, _ = self.rnn(x)
        # |x| = (batch,length,hs*2)
        y = self.activation(self.gen(x[:,-1]))
        # |y| = (bs,n_class)
        
        return y