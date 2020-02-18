from tqdm import tqdm

import torch
import pickle
import utils

import os
import Constants

import matplotlib.pyplot as plt

from vocab import Vocab
sick_vocab_file = os.path.join('data/sick', 'sick.vocab')
vocab = Vocab(filename=sick_vocab_file,
              data=[Constants.PAD_WORD, Constants.UNK_WORD,
                    Constants.BOS_WORD, Constants.EOS_WORD])


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        losses = []
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(int(len(dataset)/10)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            target = utils.map_label_to_target(label, dataset.num_classes)
            linput, rinput = linput.to(self.device), rinput.to(self.device)
            target = target.to(self.device)
            
            output, intermediate_output = self.model(ltree, linput, rtree, rinput)
            loss = self.criterion(output, target)
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        
        print(losses)
        #plt.plot(losses)
        #print(plt.show())
        exit()
        return total_loss / len(dataset)
    
    
    # helper function for testing
    def test(self, dataset, save_attention=False):
        self.model.eval()
        atten_val = []
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(int(len(dataset))), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                '''
                lin = []
                rin = []
            
                for l in linput:
                    lin.append(vocab.getLabel(int(l)))
                for r in rinput:
                    rin.append(vocab.getLabel(int(r)))

                print("##")
                print(' '.join(lin))
                print(' '.join(rin))
                
                #output = self.model(ltree, linput, rtree, rinput)
                output, intermediate_output  = self.model(ltree, linput, rtree, rinput)
                
                print("##")
                print(output.argmax().item()+1)
                '''
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
                if save_attention :
                    atten_val.append(intermediate_output)
        #if save_attention :
        #    with open("atten_val.pkl","wb") as f:
        #        pickle.dump(atten_val,f, protocol = pickle.HIGHEST_PROTOCOL)
        return total_loss / len(dataset), predictions
