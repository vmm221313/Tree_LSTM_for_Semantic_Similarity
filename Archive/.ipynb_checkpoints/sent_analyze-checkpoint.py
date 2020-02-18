
# !mkdir dst; for l in `cat best_model.txt`; do  f=`echo $l|tr ',' '_'|cut -d '_' -f1-5`; for file in `find checkpoints/  -name $f*`; do cp $file dst/; done; done

import os
import pickle
import numpy as np
import statistics as st
from sklearn.metrics import mean_squared_error 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt




data_loc = "data/sick/test"
filelist=[]
base, rand, alldep, mdep = [], [], [] ,[]

#################################################
#
# Get prediction of different trained model
#
# ################################################

target = []
with open(os.path.join("best_model.txt"),"r") as mdf:
    for lines in mdf:
        ln = lines
        name = ln.split(",")
        file = "_".join(name[:-3])
        output_file = os.path.join("dst",file+"_target_pred_.pkl" )
        with open(output_file,"rb") as outf :
            target_pred = pickle.load(outf) 
            if "base" in output_file:
                base.append(target_pred[1].numpy())
            elif "alldep" in output_file:
                alldep.append(target_pred[1].numpy())
            elif "mdep" in output_file:
                mdep.append(target_pred[1].numpy())
            elif "rand" in output_file :
                rand.append(target_pred[1].numpy())
        filelist.append(file)

target = target_pred[0].numpy()


#################################################
#
# Compute evaluation metric
#
# ################################################

def calculate_matric(target, model):
    pearson_score, spearman_score, mse_score = [], [], [] 
    
    for data in model:
        pearson_score.append(float(pearsonr(target,data)[0])) #pearson
        spearman_score.append(float(spearmanr(target,data)[0])) #"spearman":
        mse_score.append(float(mean_squared_error(target,data))) #"mse":

    print (round(st.mean(pearson_score),4), round(st.stdev(pearson_score),4),\
        round(st.mean(spearman_score),4), round(st.stdev(spearman_score),4),\
        round(st.mean(mse_score),4), round(st.stdev(mse_score),4))

calculate_matric(target, base)
calculate_matric(target, rand)
calculate_matric(target, alldep)
calculate_matric(target, mdep)

"""
(0.8649, 0.0045, 0.8058, 0.004, 0.2588, 0.0091). base
(0.8589, 0.0037, 0.7989, 0.0051, 0.2691, 0.0069) rand
(0.8614, 0.0056, 0.8016, 0.0085, 0.2662, 0.0111). alldep
(0.8667, 0.0015, 0.8077, 0.0021, 0.2545, 0.0036). mdep
"""



pred_base = np.mean(base, axis=0)
pred_rand = np.mean(rand,axis=0)
pred_alldep = np.mean(alldep,axis=0)
pred_mdep = np.mean(mdep,axis=0)



#################################################
#
# Read Test Sentences
#
# ################################################


sent_a, sent_b, sent_idx = [], [], []
length_a, length_b = [], []
i  = 0
with open(data_loc+"/a.toks") as af, open(data_loc+"/b.toks") as bf:
  for (al,bl) in zip(af.readlines(),bf.readlines()):
    s1 = al.strip()
    s2 = bl.strip()
    len_1 = len(s1.split())
    len_2 = len(s2.split())
    #if abs(len_1 - len_2) < 5:
    sent_a.append(s1)
    sent_b.append(s2)
    length_a.append(len_1)
    length_b.append(len_2)
    sent_idx.append(i)
    i+=1
     

#################################################
#
#  Analyse sentence
#
# ################################################



res1=[]
res2=[]
res3=[]

count = 0
for i,sc in enumerate(target):
    g = round(target[i],2)
    b = round(pred_base[i],2)
    m = round(pred_mdep[i],2)
    a = round(pred_alldep[i],2)
    r = round(pred_rand[i],2)
    print(i, sent_a[i], sent_b[i],g,b,m,a,r)

    if abs(g-b) > abs(g-m) and g> 2 and g<3.4:
        print(i, sent_a[i], sent_b[i],g,b,m,a,r)
        count+=1

"""
print(count)

    tok_a = set(sent_a[i].lower().split())
    tok_b = set(sent_b[i].lower().split())
    tok_b = tok_b.difference({'the','is','a','an'})
    tok_a = tok_a.difference({'the','is','a','an'})
    if len(tok_a.difference(tok_b))==1 and abs(g-b) < abs(g-m) :#or len(tok_b.difference(tok_a))<3 :
        print(i, sent_a[i], sent_b[i],g,b,m,a,r)
        count +=1
"""

    if abs(g-b)> abs(g-m) and abs(b-m)>0.3:
        print(i, sent_a[i], sent_b[i],g,b,m,a,r)
        count+=1

    print(i, sent_a[i], sent_b[i],g,b,m,a,r)
    if abs(g-b)> abs(g-m) :
        print(i, sent_a[i], sent_b[i],g,b,m,a,r)
        count+=1

    if g>=2.0 and g<=3.0 and abs(g-m)>0.3 and abs(g-b)> abs(g-m) :
        print(sent_a[i], sent_b[i],g,b,m)


  if g>2.0 and g<3.0 and abs(g-b)>0.5 and abs(g-b)> abs(g-m) :
    print(sent_a[i], sent_b[i],g,b,m)

  if abs(g-b)>1.5 : 
    print(sent_a[i], sent_b[i],g,b,m)

  #if  g==b:
  #  print(sent_a[i], sent_b[i],g,m,b)
  #mdep is better
  if abs(g-b)>0.1 and  abs(g-m)<0.05:
    res1.append([sent_a[i], sent_b[i],g,m,b])
    #print (sent_a[i], sent_b[i],g,m,b)

  #base is better
  if abs(g-b)<0.05 and  abs(g-m)>0.1:
    res2.append([sent_a[i], sent_b[i],g,m,b])
    #print (sent_a[i], sent_b[i],g,m,b)

  if abs(g-b)>2 and  abs(g-m)>0.1
print(len(res2))


################################
#
# Temp
#
# ###############################
