from algorithms.CreateTensor import readTensor
from algorithms.tensor_coclust_incremental import CoClust
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import numpy as np


dataset = 'DBLP'   #DBLP, MovieLens1,MovieLens2, yelpTOR, yelpPGH


data_path = './datasets/tensors/'
T, target, label_mode = readTensor(dataset, data_path)
    

model = CoClust(verbose = True)
model.fit(T)
n = nmi(target, model.labels_[label_mode])
a = ari(target, model.labels_[label_mode])

print(f"nmi: {n}")
print(f"ari: {a}")
