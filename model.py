import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size,batch_size=16 ,n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size=batch_size
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True,bidirectional=True)
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers*2, input.size(0), self.hidden_size))
        return (hidden,cell) # hidden states and cell states
     
    def forward(self, input,input_masking):
        """
        input : B= BATCH_SIZE=16, T=LENGTH=50 (LongTensor)
        input_masking : B,T (PAD Masked ByteTensor)
        
        Real Context except <PAD> is recreated as output
        """
        
        self.hidden = self.init_hidden(input)
        
        embedded = self.embedding(input)
        output, self.hidden = self.lstm(embedded, self.hidden) #output.shape = (B,LENGTH,EMB_SIZE)
        
        decoder_starter=[]
        
        for i,o in enumerate(output): # B,T,D
            decoder_starter.append(o[0])
            
        return output, torch.cat(decoder_starter).view(input.size(0),-1).unsqueeze(1)

    
class Decoder(nn.Module):
    
    def __init__(self,slot_size,intent_size,embedding_size,hidden_size,batch_size=16,n_layers=1,dropout_p=0.5):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size+self.hidden_size*2, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size,self.hidden_size) # Attention
        self.slot_out = nn.Linear(self.hidden_size*2, self.slot_size) #input: h(i), s(i-1)
        self.intent_out = nn.Linear(self.hidden_size*2,self.intent_size) #input: c_intent + decoder_hidden_state // output: intent_tag
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        #self.out.bias.data.fill_(0)
        #self.out.weight.data.uniform_(-0.1, 0.1)
        #self.lstm.weight.data.
    
    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D ===> decoder_hidden : s
        encoder_outputs : B,T,D
        encoder_maskings : B,T # BoolTensor
        """
        
        hidden = hidden.squeeze(0).unsqueeze(2)  # Hidden: (1, batch, dimension) -> (placement, dimension, 1)
        
        batch_size = encoder_outputs.size(0) # B
        max_len = encoder_outputs.size(1) # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size*max_len,-1)) # B*T,D -> B*T,D
        energies = energies.view(batch_size,max_len,-1) # B,T,D (batch, time, dimension)
        attn_energies = energies.bmm(hidden).transpose(1,2) # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings,-1e12) # PAD masking

        
        alpha = F.softmax(attn_energies) # B,T
        alpha = alpha.unsqueeze(1) # B,1,T
        context = alpha.bmm(encoder_outputs) # B,1,T * B,T,D => B,1,D
        
        return context # B,1,D
    
    def init_hidden(self,input):
        hidden = Variable(torch.zeros(self.n_layers*1,input.size(0), self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers*1, input.size(0), self.hidden_size))
        return (hidden,cell)
    
    def forward(self, input,context,encoder_outputs,encoder_maskings,training=True):
        """
        input : B,1(length) ====> start_decode.shape(16,1): word2index['<SOS>'] = 2
        context : B,1,D = 16,1,128 ====> (at first step: real context hidden state in encoder) h(last)
        
        """
        
        # Get the embedding of the current input word
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode=[]
        aligns = encoder_outputs.transpose(0,1)
        length = encoder_outputs.size(1) # length
        # intent detection
        intent_hidden = hidden[0].clone()
        intent_context = self.Attention(intent_hidden, encoder_outputs,encoder_maskings)
        concated = torch.cat((intent_hidden,intent_context.transpose(0,1)),2) # 1,B,D
        intent_score = self.intent_out(concated.squeeze(0))
        
        for i in range(length): # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)# B,1,D # aligned.shape = torch.Size([16, 1, 128]) (h[i])
        
            _, hidden = self.lstm(torch.cat((embedded,context,aligned),2), hidden) # input, context, aligned encoder hidden, hidden
            # context = c[i] (at first: last hidden state of encoder), embedded = y[i], aligned = h[i] ===> to calculate next y[i]
            # hidden = s[i-1], decoder_cell[i-1]
            # self.lstm returns decoder_output (_) and decoder's (hidden states+ cell states)
            # hidden here is a tuple consisting of hidden and cell states
            # hidden[0].shape = torch.Size([1, 16, 128]) = decoder_hidden_state: s(i)
            # hidden[1].shape = torch.Size([1, 16, 128]) = decoder_cell_state 


            concated = torch.cat((hidden[0],context.transpose(0,1)),2) #context = (at first: h(last) from encoder)
            # concated.shape = torch.Size([1, 16, 256])
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            _,input = torch.max(softmaxed,1)
            embedded = self.embedding(input.unsqueeze(1))
            
            context = self.Attention(hidden[0], encoder_outputs,encoder_maskings) 
        
        slot_scores = torch.cat(decode,1)
        return slot_scores.view(input.size(0)*length,-1), intent_score