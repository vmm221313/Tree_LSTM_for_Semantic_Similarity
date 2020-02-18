import torch
import torch.nn as nn
import torch.nn.functional as F

import Constants


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
        #self.dropout = nn.Dropout(0.5)
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
            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

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

    def forward(self, tree, inputs, hidden_states):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, hidden_states)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
            

        tree.state = self.node_forward(inputs[tree.idx], tree.rel, child_c, child_h)
    
        if tree.num_children != 0:
            hidden_states.append(tree.state[1].tolist())
               
        return tree.state, hidden_states


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes, out_dim):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.wh = nn.Linear(2 * self.out_dim*self.out_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, lvec, rvec):
     
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = torch.sigmoid(self.wh(vec_dist))
        out = self.dropout(out)
        out = F.log_softmax(self.wp(out), dim=1)
        
        #shape of output should be [1,5]
        return out


class Capsule(nn.Module):
    def __init__(self, hidden_dim, out_dim, mem_dim):
        super(Capsule, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mem_dim = mem_dim
        self.softmax = nn.Softmax(dim=1)
        self.b_init=torch.zeros(20, out_dim)
        self.w_tj = nn.Linear(self.mem_dim, self.out_dim, bias=True)
        
    def forward(self, num_nodes, hidden_states):
        b=self.b_init[:num_nodes]
        c = self.softmax(b)
        
        hidden_states = torch.tensor(hidden_states)
        h_jt = self.w_tj(hidden_states)
    
        v = torch.mm(h_jt.T,c) 
        mod_v= torch.sqrt(torch.sum(v*v,axis=0))
        sentence_vector= mod_v*v/(1+mod_v*mod_v)

        b = b + torch.mm(h_jt, sentence_vector )
        c = self.softmax(b)

        return sentence_vector


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, model_type, rel_vocab, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(model_type, rel_vocab, in_dim, mem_dim)
        out_dim = 100
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes, out_dim)
        #self.similarity = Similarity(mem_dim, hidden_dim, num_classes)
        self.num_classes = num_classes
        
        self.capsule = Capsule(hidden_dim, out_dim, mem_dim)

    def forward(self, ltree, linputs, rtree, rinputs):
        
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        (lstate, lhidden), all_lhidden = self.childsumtreelstm(ltree, linputs, [])
        (rstate, rhidden), all_rhidden = self.childsumtreelstm(rtree, rinputs, [])
        
        
        all_lhidden = [hidden_state[-1] for hidden_state in all_lhidden]
        all_rhidden = [hidden_state[-1] for hidden_state in all_rhidden]

        lsent = self.capsule(len(all_lhidden), all_lhidden)
        rsent = self.capsule(len(all_rhidden), all_rhidden)
    
        lsent = lsent.view(1,-1)
        rsent = rsent.view(1,-1)
        output = self.similarity(lsent, rsent)
        
        #output = self.similarity(lhidden, rhidden)   
        
        return output,["None"]


