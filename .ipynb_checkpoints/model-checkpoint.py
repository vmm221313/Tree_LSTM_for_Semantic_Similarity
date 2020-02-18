import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import Vocab
from config import parse_args

global args
args = parse_args()

# %tb
import bert
import Constants
import re
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Attention(nn.Module):
    def __init__(self, Matrix, hidden_dim, num_children):
        super(Attention, self).__init__()
        self.Matrix=Matrix
        self.hidden_dim=hidden_dim
        self.num_children=num_children
        self.key=nn.Linear(self.num_children,self.num_children,bias=False)
        self.query=nn.Linear(self.num_children,self.num_children,bias=False)
        self.value=nn.Linear(self.num_children,self.num_children,bias=False)
        self.softmax = nn.Softmax(dim=1)
        torch.nn.init.xavier_uniform_(self.key.weight)
        torch.nn.init.xavier_uniform_(self.query.weight)
        torch.nn.init.xavier_uniform_(self.value.weight)
        
    def forward(self):
        #shape of matrix is (num_children,h_dim)
        #shape of query key n value is (h_dim,num_children)
        query = self.query(self.Matrix)
        key = self.key(self.Matrix)
        value = self.value(self.Matrix)
        align = torch.mm(query.T,key)/np.sqrt(self.hidden_dim)
        alpha = self.softmax(align)
        #hidden_state = torch.bmm(alpha,value)                   paper said this but isnt correct????
        hidden_state = torch.mm(alpha,value.T)
        return hidden_state

#set(relations.values())
# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, model_type, rel_vocab, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.rel_vocab = rel_vocab
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.rel_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.rel_mat = {}
        self.dropout = nn.Dropout(0.2)
        self.model_type = model_type.lower()

        # if model_type == "mdep" : #merged
        #     for k in  set(merged_relations.values()) :
        #         self.rel_mat['rel_mat_'+str(k)] = nn.Linear(self.mem_dim, self.rel_dim, bias=False) 
        #     self.rel_mat['rel_mat_other'] = nn.Linear(self.mem_dim, self.rel_dim, bias=False)
        #     self.rel_mat = nn.ModuleDict(self.rel_mat)

        # elif model_type == "alldep":
        #     for k in rel_vocab.labelToIdx.keys():
        #         self.rel_mat['rel_mat_'+str(k)] = nn.Linear(self.mem_dim, self.rel_dim, bias=False) 
        #         #self.rel_mat['rel_mat_'+str(k)].apply(self.init_weights) 
        #     self.rel_mat['rel_mat_other'] = nn.Linear(self.mem_dim, self.rel_dim, bias=False)
        #     self.rel_mat = nn.ModuleDict(self.rel_mat)
        
        if self.model_type == "rand":
            for k in rel_vocab.labelToIdx.keys():
                self.rel_mat['rel_mat_'+str(k)] = nn.Linear(self.mem_dim, self.rel_dim, bias=False) 
            self.rel_mat['rel_mat_other'] = nn.Linear(self.mem_dim, self.rel_dim, bias=False)
        else :
            if self.model_type == "mdep" : #merged
                for k in  set(Constants.merged_relations.values()) :
                    self.rel_mat['rel_mat_'+str(k)] = nn.Linear(self.mem_dim, self.rel_dim, bias=False) 
            
            elif self.model_type == "alldep":
                for k in rel_vocab.labelToIdx.keys():
                    self.rel_mat['rel_mat_'+str(k)] = nn.Linear(self.mem_dim, self.rel_dim, bias=False) 
                    #self.rel_mat['rel_mat_'+str(k)].apply(self.init_weights) 
            self.rel_mat['rel_mat_other'] = nn.Linear(self.mem_dim, self.rel_dim, bias=False)
            self.rel_mat = nn.ModuleDict(self.rel_mat)

            '''
            for k in rel_vocab.labelToIdx.viewkeys():
                self.rel_mat = nn.ModuleDict({
                    'rel_mat_'+str(k): nn.Linear(self.mem_dim, self.mem_dim) 
                })
            '''

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            #m.bias.data.fill_(0.01)


    def node_forward(self, inputs, relations, child_c, child_h):


        if self.model_type == "mdep" :
            relation_type = Constants.merged_relations.get(relations,'other')
        elif self.model_type == "alldep" or self.model_type == "rand":
            relation_type = relations

        if self.model_type != "base":
            if relation_type == 'root' : 
                child_h_rel = child_h
            else :
                child_h_rel = self.rel_mat['rel_mat_'+relation_type](child_h)
            child_h_sum = torch.sum(child_h_rel, dim=0, keepdim=True)
        else :
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)           #child_h is list of all 
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))

        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)
            

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            #write attention code here
            obj = Attention(child_h.T,child_h.shape[1],child_h.shape[0])
            child_h = obj.forward()
            #child_h will be ur output
        tree.state = self.node_forward(inputs[tree.idx], tree.rel, child_c, child_h)
        
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = torch.sigmoid(self.wh(vec_dist))
        out = self.dropout(out)
        out = F.log_softmax(self.wp(out), dim=1)

        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, model_type, rel_vocab, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)         
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(model_type, rel_vocab, in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
        self.sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        self.vocab = Vocab(filename=self.sick_vocab_file,data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD,        Constants.EOS_WORD])

    def forward(self, ltree, linputs, rtree, rinputs):
        #linputs = self.emb(linputs)    #emb passed thru main
        #rinputs = self.emb(rinputs)
        linputs, rinputs = bert.get_bert_embd(linputs,rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs.float())
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs.float())
        #output = self.similarity(lstate, rstate)
        output = self.similarity(lhidden, rhidden)
        return output,["None"]
