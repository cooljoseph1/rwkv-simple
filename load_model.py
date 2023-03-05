from typing import Dict
import os
import urllib.request
from tqdm import tqdm
from my_model import myRWKV

def RWKV(path) -> myRWKV:
    if ("http" in path):
        fileName = path.split("/")[-1]
        if not os.path.exists(fileName):
            urllib.request.urlretrieve(path, fileName)
        path = fileName

    weights, n_layers = loadWeights(path)
    return myRWKV(weights, n_layers)

def loadWeights(path):
    import torch
    n_layer = 0

    w: Dict[str, torch.Tensor] = torch.load(
        path, map_location="cpu")
    # refine weights
    keys = list(w.keys())
    for x in keys:
        w[x].requires_grad = False

        try:
            if (int(x.split('.')[1])+1 > n_layer):
                n_layer = int(x.split('.')[1])+1
        except:
            pass

    # store weights in self.w

    keys = list(w.keys())
    for x in keys:

        if '.time_' in x:
            w[x] = w[x].squeeze()

        if '.time_decay' in x:
            w[x] = -torch.exp(w[x].double())

        if 'receptance.weight' in x:
            w[x] = w[x]

    # Transform Weights from backend
    for x in tqdm(list(w.keys())):
        if "emb.weight" in x:
            w[x] = w[x].squeeze().to("cpu")
    return w, n_layer