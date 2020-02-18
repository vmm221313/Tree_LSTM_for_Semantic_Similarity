import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', help ="name of input file")
args = parser.parse_args()
infile = args.input


#infile ="2019-9-16-2-50_300_150_50_adagrad_0.05_0.0001_base_target_pred_.pkl"
#infile = "baseline/2019-9-14-3-21_300_150_50_adagrad_0.025_0.0001.log"

train_pearson=[]
train_spearman=[]
train_mse=[]
train_loss=[]

dev_pearson=[]
dev_spearman=[]
dev_mse=[]
dev_loss=[]

test_pearson=[]
test_spearman=[]
test_mse=[]
test_loss=[]


with open(infile, "r") as f:
    for lines in f:
        line = lines.lower()
        epoch_start = line.find('epoch')
        epoch_end = line.find(',',int(epoch_start))
        epoch = int(line[epoch_start+5:epoch_end ])
        if 'test' in line : 
            data = line.strip().split("\t")
            for dt in data:
                if 'loss' in dt:
                    tmp = dt.split(":")
                    test_loss.insert(epoch,float(tmp[1].strip()))
                elif 'pearson' in dt:
                    tmp = dt.split(":")
                    test_pearson.insert(epoch,float(tmp[1].strip()))
                elif 'spearman' in dt:
                    tmp = dt.split(":")
                    test_spearman.insert(epoch,float(tmp[1].strip()))
                elif 'mse' in dt:
                    tmp = dt.split(":")
                    test_mse.insert(epoch,float(tmp[1].strip()))
        elif 'train' in line : 
            data = line.strip().split("\t")
            for dt in data:
                if 'loss' in dt:
                    tmp = dt.split(":")
                    train_loss.insert(epoch,float(tmp[1].strip()))
                elif 'pearson' in dt:
                    tmp = dt.split(":")
                    train_pearson.insert(epoch,float(tmp[1].strip()))
                elif 'spearman' in dt:
                    tmp = dt.split(":")
                    train_spearman.insert(epoch,float(tmp[1].strip()))
                elif 'mse' in dt:
                    tmp = dt.split(":")
                    train_mse.insert(epoch,float(tmp[1].strip()))

        elif 'dev' in line : 
            data = line.strip().split("\t")
            for dt in data:
                if 'loss' in dt:
                    tmp = dt.split(":")
                    dev_loss.insert(epoch,float(tmp[1].strip()))
                elif 'pearson' in dt:
                    tmp = dt.split(":")
                    dev_pearson.insert(epoch,float(tmp[1].strip()))
                elif 'spearman' in dt:
                    tmp = dt.split(":")
                    dev_spearman.insert(epoch,float(tmp[1].strip()))
                elif 'mse' in dt:
                    tmp = dt.split(":")
                    dev_mse.insert(epoch,float(tmp[1].strip()))
        


def plot_figure(train_data, dev_data, test_data, ylab, fname):
    x = np.linspace(1,20,20)
    plt.plot(x, train_data, label="Train")
    plt.plot(x, dev_data, label="Dev")
    plt.plot(x, test_data, label="Test") 
    plt.xticks(np.arange(0, 21, step=2))
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.legend(loc='upper left')
    plt.savefig(fname)
    plt.close()


plot_figure(train_pearson,dev_pearson,test_pearson, ylab="Pearson's r", fname="pearson.jpg")
plot_figure(train_spearman,dev_spearman,test_spearman, ylab="Spearman's r", fname="spearman.jpg")
plot_figure(train_mse,dev_mse,test_mse, ylab="MSE", fname="mse.jpg")
plot_figure(train_loss,dev_loss,test_loss, ylab="Loss", fname="loss.jpg")


