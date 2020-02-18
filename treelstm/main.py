from __future__ import division
from __future__ import print_function
import warnings
import pandas as po


import os
import random
import logging
import pickle
from tqdm import tqdm_notebook

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SICKDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
import utils
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer
# CONFIG PARSER
from config import parse_args
from scipy.stats import pearsonr, spearmanr
import numpy as np
from datetime import datetime as dt

global args
args = parse_args()
# global logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
# file logger

dateformat = "-".join([str(x) for x in [dt.today().year, dt.today().month, \
    dt.today().day, dt.today().hour, dt.today().minute ]])

dateformat

logfile_name = "_".join([str(x) for x in [dateformat, args.input_dim, args.mem_dim, \
    args.hidden_dim, args.optim, args.lr, args.wd, args.batchsize, args.model_type]])

chkptloc = args.save+args.model_type+"/"


if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists(chkptloc):
    os.makedirs(chkptloc)


if (args.load_model):
    logfile_name = args.saved_model 
    fh = logging.FileHandler(os.path.join(args.save, logfile_name )+'.log', mode='a')

else :
    fh = logging.FileHandler(os.path.join(chkptloc, logfile_name )+'.log', mode='w')

fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

# console logger
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# argument validation
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
if args.sparse and args.wd != 0:
    logger.error('Sparsity and weight decay are incompatible, pick one!')
    exit()


logger.debug(args)
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#     torch.backends.cudnn.benchmark = True


train_dir = os.path.join(args.data, 'train/')
dev_dir = os.path.join(args.data, 'dev/')
test_dir = os.path.join(args.data, 'test/')

# write unique words from all token files
sick_vocab_file = os.path.join(args.data, 'sick.vocab')
if not os.path.isfile(sick_vocab_file):
    token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
    token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
    token_files = token_files_a + token_files_b
    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    utils.build_vocab(token_files, sick_vocab_file)

#ADDED
sick_rel_file = os.path.join(args.data, 'sick.rel')
if not os.path.isfile(sick_rel_file):
    token_files_b = [os.path.join(split, 'b.rels') for split in [train_dir, dev_dir, test_dir]]
    token_files_a = [os.path.join(split, 'a.rels') for split in [train_dir, dev_dir, test_dir]]
    token_files = token_files_a + token_files_b
    sick_rel_file = os.path.join(args.data, 'sick.rel')
    utils.build_vocab(token_files, sick_rel_file)

# get vocab object from vocab file previously written
vocab = Vocab(filename=sick_vocab_file,
              data=[Constants.PAD_WORD, Constants.UNK_WORD,
                    Constants.BOS_WORD, Constants.EOS_WORD])
logger.debug('==> SICK vocabulary size : %d ' % vocab.size())

#ADDED
#get relation vocab object from relation
rel_vocab = Vocab(filename=sick_rel_file)
logger.debug('==> RELATION vocabulary size : %d ' % rel_vocab.size())


# load SICK dataset splits
train_file = os.path.join(args.data, 'sick_train.pth')
if os.path.isfile(train_file):
    train_dataset = torch.load(train_file)
else:
    train_dataset = SICKDataset(train_dir, vocab, rel_vocab, args.num_classes)
    torch.save(train_dataset, train_file)
logger.debug('==> Size of train data   : %d ' % len(train_dataset))


dev_file = os.path.join(args.data, 'sick_dev.pth')
if os.path.isfile(dev_file):
    dev_dataset = torch.load(dev_file)
else:
    dev_dataset = SICKDataset(dev_dir, vocab, rel_vocab, args.num_classes)
    torch.save(dev_dataset, dev_file)
logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))



test_file = os.path.join(args.data, 'sick_test.pth')
if os.path.isfile(test_file):
    test_dataset = torch.load(test_file)
else:
    test_dataset = SICKDataset(test_dir, vocab, rel_vocab, args.num_classes)
    torch.save(test_dataset, test_file)
logger.debug('==> Size of test data     : %d ' % len(test_dataset))



# initialize model, criterion/loss_function, optimizer
model = SimilarityTreeLSTM(
    args.model_type,
    rel_vocab,
    vocab.size(),
    args.input_dim,
    args.mem_dim,
    args.hidden_dim,
    args.num_classes,
    args.sparse,
    args.freeze_embed,)

criterion = nn.KLDivLoss()
#criterion = nn.CrossEntropyLoss()





# +
###For changing embeddings
# -

# for words common to dataset vocab and GLOVE, use GLOVE vectors
# for other words in dataset vocab, use random normal vectors
emb_file = os.path.join(args.data, 'sick_embed.pth')
if os.path.isfile(emb_file):
    emb = torch.load(emb_file)
else:
    # load glove embeddings and vocab
    print("embedding")
    glove_vocab, glove_emb = utils.load_word_vectors(
        os.path.join(args.glove, 'glove.840B.300d')) #glove.840B.300d
    logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
    emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
    emb.normal_(0, 0.05)
    # zero out the embeddings for padding and other special words if they are absent in vocab
    for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                Constants.BOS_WORD, Constants.EOS_WORD]):
        emb[idx].zero_()
    for word in vocab.labelToIdx.keys():
        if glove_vocab.getIndex(word):
            emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
    torch.save(emb, emb_file)
# plug these into embedding matrix inside model
model.emb.weight.data.copy_(emb)

# +
###For changing embeddings
# -





# +
model.to(device), criterion.to(device)
if args.optim == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'adagrad':
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
elif args.optim == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                 model.parameters()), lr=args.lr, weight_decay=args.wd)

elif args.optim == 'rmsprop':
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=args.lr, weight_decay=args.wd)
# -

metrics = Metrics(args.num_classes)

# create trainer object for training and testing
if (args.load_model):
    model_ckpt = torch.load('%s.pt' % os.path.join(args.save, args.saved_model))
    model.load_state_dict(model_ckpt['model'])
    trainer = Trainer(args, model, criterion, optimizer, device)
else :
    trainer = Trainer(args, model, criterion, optimizer, device)

# whether to continue training or only evaluate
if (args.evaluate):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        predictions = torch.zeros(len(test_dataset), dtype=torch.float, device='cpu')
        indices = torch.arange(1, test_dataset.num_classes + 1, dtype=torch.float, device='cpu')
        for idx in tqdm_notebook(range(len(test_dataset)), desc='Testing epoch  ' + str(args.epochs) + ''):
            ltree, linput, rtree, rinput, label = test_dataset[idx]
            target = utils.map_label_to_target(label, test_dataset.num_classes)
            linput, rinput = linput.to(device), rinput.to(device)
            target = target.to(device)
            #output = self.model(ltree, linput, rtree, rinput)
            output, intermediate_output  = model(ltree, linput, rtree, rinput)
            loss = criterion(output, target)
            total_loss += loss.item()
            output = output.squeeze().to('cpu')
            predictions[idx] = torch.dot(indices, torch.exp(output))
    test_loss, test_pred = trainer.test(test_dataset)
    test_pearson = metrics.pearson(test_pred, test_dataset.labels)
    test_mse = metrics.mse(test_pred, test_dataset.labels)
    test_spear = (spearmanr(np.asarray(test_pred), np.asarray(test_dataset.labels)))[0]
    print (" Test \tLoss: {}\tPearson: {}\t spearman:{}\t MSE: {}".format(
        args.epochs, test_loss, test_pearson, test_spear, test_mse))
    with open("predictions.pkl","wb") as f:
            pickle.dump([test_pred, test_dataset.labels],f, protocol = pickle.HIGHEST_PROTOCOL)

else :
    best = -float('inf')
    epoch_count = 0
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)
        dev_loss, dev_pred = trainer.test(dev_dataset)
        test_loss, test_pred = trainer.test(test_dataset)


        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        train_spear = (spearmanr(np.asarray(train_pred), np.asarray(train_dataset.labels)))[0]
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\t spearman:{}\t MSE: {}'.format(
            epoch, train_loss, train_pearson, train_spear, train_mse))
        

        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        dev_spear = (spearmanr(np.asarray(dev_pred), np.asarray(dev_dataset.labels)))[0]
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\t spearman:{}\t MSE: {}'.format(
            epoch, dev_loss, dev_pearson, dev_spear, dev_mse))

        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        test_spear = (spearmanr(np.asarray(test_pred), np.asarray(test_dataset.labels)))[0]

        logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\t spearman:{}\t MSE: {}'.format(
            epoch, test_loss, test_pearson, test_spear, test_mse))

        if best < dev_pearson:

            best = dev_pearson 

            save_test_pearson = test_pearson
            save_test_spear =  test_spear       
            save_test_mse = test_mse

            if args.model_type == "mdep":
                m_rel = Constants.merged_relations
            elif args.model_type == "alldep":
                m_rel = "all"
            else :
                m_rel = None
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson,'spearman' : test_spear, 'mse': test_mse, 
                'args': args, 'epoch': epoch, "rel_mat" : m_rel,
            }
            #Save predictions

            pred_file = os.path.join(chkptloc, logfile_name+"_target_pred_.pkl")
            with open(pred_file,"wb") as f:
                pickle.dump([test_dataset.labels,test_pred],f, protocol = pickle.HIGHEST_PROTOCOL)
            #test_loss, test_pred = trainer.test(test_dataset,save_attention=True)
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(chkptloc, logfile_name))
    with open(os.path.join(chkptloc,"result_"+args.model_type+".csv"),"a") as f:
        res = logfile_name.replace("_",",")
        res=res+","+str(round(float(save_test_pearson),4))+","+str(round(float(save_test_spear),4))\
            +","+str(round(float(save_test_mse),4))+"\n"
        f.writelines(res)
        
    with open(os.path.join(chkptloc,"Results"+".csv"),"a") as f:
        #res = logfile_name.replace("_",",")
        res=str(args.hidden_dim)+","+str(args.mem_dim)+","+str(args.lr)+","+str(round(float(save_test_pearson),4))+","+str(round(float(save_test_spear),4))\
            +","+str(round(float(save_test_mse),4))+"\n"
        f.writelines(res) 

  

# results = po.read_csv(os.path.join('Results.csv'))
#     dict = {'hidden_dim': args.hidden_dim,
#             'lr': args.lr,
#             'pearson': round(float(save_test_pearson),4),
#             'spearman': round(float(save_test_spear),4),
#             'mse': round(float(save_test_mse),4),
#             'mem_dim': args.mem_dim
#        }
#
#     results.append(dict, ignore_index=True)
#     results.to_csv('Results.csv')
#     

# #python main.py --lr 0.05 --wd 0.0001 --optim adagrad --batchsize 25 --freeze_embed --epochs 30
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
#     # data arguments
#     parser.add_argument('--data', default='data/sick/',
#                         help='path to dataset')
#     parser.add_argument('--glove', default='data/glove/',
#                         help='directory with GLOVE embeddings')
#     parser.add_argument('--save', default='checkpoints/',
#                         help='directory to save checkpoints in')
#     parser.add_argument('--expname', type=str, default='test',
#                         help='Name to identify experiment')
#     # model arguments
#     #parser.add_argument('--model_type', required=True, help ='type of model')
#     parser.add_argument('--model_type', default="mdep", help ='type of model')
#     parser.add_argument('--input_dim', default=300, type=int,
#                         help='Size of sparse word vector')
#     parser.add_argument('--mem_dim', default=150, type=int, #150
#                         help='Size of TreeLSTM cell state')
#     parser.add_argument('--hidden_dim', default=50, type=int, #50
#                         help='Size of classifier MLP')
#     parser.add_argument('--num_classes', default=5, type=int,
#                         help='Number of classes in dataset')
#     parser.add_argument('--freeze_embed', action='store_false',
#                         help='Freeze word embeddings')
#     parser.add_argument('--load_model', action='store_true', #ADDED
#                         help='Load a saved model')
#     parser.add_argument('--evaluate', action='store_true', #ADDED
#                         help='evaluate a model')
#     parser.add_argument('--saved_model', help='Name of saved model') 
#
#     # training arguments
#     parser.add_argument('--epochs', default=20, type=int,
#                         help='number of total epochs to run')
#     parser.add_argument('--batchsize', default=25, type=int,
#                         help='batchsize for optimizer updates')
#     parser.add_argument('--lr', default=0.05, type=float, #0.01
#                         metavar='LR', help='initial learning rate')
#     parser.add_argument('--wd', default=1e-4, type=float,
#                         help='weight decay (default: 1e-4)')
#     parser.add_argument('--sparse', action='store_true',
#                         help='Enable sparsity for embeddings, \
#                               incompatible with weight decay')
#     parser.add_argument('--optim', default='adagrad',
#                         help='optimizer (default: adagrad)')
#     # miscellaneous options
#     parser.add_argument('--seed', default=123, type=int,
#                         help='random seed (default: 123)')
#     cuda_parser = parser.add_mutually_exclusive_group(required=False)
#     cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
#     cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
#     parser.set_defaults(cuda=True)
#
#     args = parser.parse_args()
#     return args
