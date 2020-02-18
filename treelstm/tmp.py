import random

args.save_model = "2019-9-22-11-27_300_150_50_adagrad_0.05_0.005_25_mdep"
data_loc = "../data/sick/test"

sent_a=[]
sent_b=[]


with open(data_loc+"/a.toks") as af, open(data_loc+"/b.toks") as bf:
  for (al,bl) in zip(af.readlines(),bf.readlines()):
    s1 = al.strip()
    s2 = bl.strip()
    sent_a.append(s1)
    sent_b.append(s2)

n_selections = 500
randIndex = random.sample(range(len(test_dataset)), n_selections)
#randIndex.sort()

#ltree_const, linput_const, rtree, rinput, label = test_dataset[randIndex[5]]
first_sent_index = 2087 #1652
first_sent_index = 1652
ltree_const, linput_const, rtree, rinput, label = test_dataset[first_sent_index]
model.eval()
with torch.no_grad():
    total_loss = 0.0
    predictions = torch.zeros(len(test_dataset), dtype=torch.float, device='cpu')
    indices = torch.arange(1, test_dataset.num_classes + 1, dtype=torch.float, device='cpu')
    #for idx in range(len(test_dataset)):
    for i, idx in enumerate(randIndex):
        ltree, linput, rtree, rinput, label = test_dataset[idx]
        #linput, rinput = linput.to(device), rinput.to(device)
        output, intermediate_output  = model(ltree_const, linput_const, rtree, rinput)
        output = output.squeeze().to('cpu')
        predictions [idx] = round(torch.dot(indices, torch.exp(output)),2)

predictions =predictions.numpy()

x={}
sorted_idx = np.argsort(-predictions)[:100]
first_sent = sent_a[first_sent_index]
for i,idx in enumerate(sorted_idx):
    x [sent_b[idx]] = predictions[idx]

sorted_x = sorted(x.items(), key=lambda kv: kv[1])
sorted_x.reverse()


