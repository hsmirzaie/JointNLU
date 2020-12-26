import os
import torch
from torch.autograd import Variable

def load_data(path):
    
    data_raw = open(path,"r").readlines()
    data_raw = [t[:-1] for t in data_raw]
    data_raw = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in data_raw]
    data_raw = [[t[0][1:-1],t[1][1:],t[2]] for t in data_raw]
    
    return data_raw

def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix['<UNK>'], seq))
    tensor = Variable(torch.LongTensor(idxs))
    return tensor

def process_data(data_raw, LENGTH):
        
    seq_input, seq_slots, intents = list(zip(*data_raw))

    flatten = lambda l: [item for sublist in l for item in sublist]
    
    vocab = set(flatten(seq_input))
    slot_tag = set(flatten(seq_slots))
    intent_tag = set(intents)
    
    padded_input = []
    padded_slots = []
    
    for i in range(len(seq_input)):
        temp = seq_input[i].copy()
        if len(temp)<LENGTH:
            temp.append('<EOS>')
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
            temp[-1]='<EOS>'
        padded_input.append(temp)
    
        temp = seq_slots[i].copy()
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
            temp[-1]='<EOS>'
        padded_slots.append(temp)
    
    
    
    word2index = {'<PAD>': 0, '<UNK>':1,'<SOS>':2,'<EOS>':3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token]=len(word2index)

    
    slot2index = {'<PAD>' : 0, '<UNK>': 1}
    for tag in slot_tag:
        if tag not in slot2index.keys():
            slot2index[tag] = len(slot2index)
            
            
    intent2index={'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)
            
    data_padded = list(zip(padded_input,padded_slots, intents))
    
    data_processed = []
    
    for d in data_padded:
        temp = prepare_sequence(d[0],word2index)
        temp = temp.view(1,-1)

        temp2 = prepare_sequence(d[1],slot2index)
        temp2 = temp2.view(1,-1)
        
        temp3 = Variable(torch.LongTensor([intent2index[d[2]] if d[2] in intent2index.keys() else intent2index['<UNK>']]))
        
        data_processed.append((temp,temp2,temp3))
       
    return data_processed, word2index, slot2index, intent2index
    
    
def test_process(test_raw, word2index, slot2index, intent2index, LENGTH):
    
    seq_input, seq_slots, intents = list(zip(*test_raw))

    flatten = lambda l: [item for sublist in l for item in sublist]
    
    padded_input = []
    padded_slots = []
    
    for i in range(len(seq_input)):
        temp = seq_input[i].copy()
        if len(temp)<LENGTH:
            temp.append('<EOS>')
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
            temp[-1]='<EOS>'
        padded_input.append(temp)
    
        temp = seq_slots[i].copy()
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
            temp[-1]='<EOS>'
        padded_slots.append(temp)
        
    data_padded = list(zip(padded_input,padded_slots, intents))
    
    data_processed = []
    
    for d in data_padded:
        temp = prepare_sequence(d[0],word2index)
        temp = temp.view(1,-1)

        temp2 = prepare_sequence(d[1],slot2index)
        temp2 = temp2.view(1,-1)
        

        temp3 = Variable(torch.LongTensor([intent2index[d[2]] if d[2] in intent2index.keys() else intent2index['<UNK>']]))
        
        data_processed.append((temp,temp2,temp3))
        
    return data_processed

    
def getBatch(batch_size,data, Shuffle = True):
    
    if Shuffle:
        random.shuffle(data)
    sindex=0
    eindex=batch_size
    while eindex < len(data):
        batch = data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        
        yield batch