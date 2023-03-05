import torch
from typing import Dict
from tqdm import tqdm

import torch.nn as nn
from torch.nn import functional as F


def matvec(x, y):
    return torch.matmul(x, y.t()).t()

def lenn(x):
    return x.shape[0]

def postProcessTensor(x):
    return x.float().cpu()

class myRWKV(nn.Module):
    def __init__(self, w: Dict[str, torch.Tensor], n_layers):
        """
        args: {
            w: idk--the saved weights?
            n_layers: The number of layers of transformers in the network
        }
        """
        super().__init__()

        self.embed_size = len(w[f"blocks.0.ffn.time_mix_k"].squeeze())
        self.n_layers = n_layers

        self.postprocess0: torch.Tensor = (w["ln_out.weight"])
        self.postprocess1: torch.Tensor = (w["ln_out.bias"])
        self.postprocess2: torch.Tensor = (w["head.weight"])
        self.emb: torch.Tensor = w["emb.weight"]
        self.emb1: torch.Tensor = w["blocks.0.ln0.weight"]
        self.emb2: torch.Tensor = w["blocks.0.ln0.bias"]
        self.ln1w: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.ln1.weight"] for x in range(n_layers)])
        self.ln1b: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.ln1.bias"] for x in range(n_layers)])
        self.ln2w: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.ln2.weight"] for x in range(n_layers)])
        self.ln2b: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.ln2.bias"] for x in range(n_layers)])
        self.time_decay: torch.Tensor = torch.stack([
            w[f"blocks.{x}.att.time_decay"] for x in range(n_layers)])
        self.time_first: torch.Tensor = torch.stack([
            w[f"blocks.{x}.att.time_first"] for x in range(n_layers)])
        self.kktk: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.att.time_mix_k"] for x in range(n_layers)])
        self.vvtv: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.att.time_mix_v"] for x in range(n_layers)])
        self.rrtr: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.att.time_mix_r"] for x in range(n_layers)])
        self.key: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.att.key.weight"] for x in range(n_layers)])
        self.value: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.att.value.weight"] for x in range(n_layers)])
        self.receptance: torch.Tensor = torch.stack([
            w[f"blocks.{x}.att.receptance.weight"] for x in range(n_layers)])
        self.outputvv: torch.Tensor = torch.stack([
            w[f"blocks.{x}.att.output.weight"] for x in range(n_layers)])
        self.time_mix_k_ffn: torch.Tensor = torch.stack([
            w[f"blocks.{x}.ffn.time_mix_k"] for x in range(n_layers)])
        self.time_mix_r_ffn: torch.Tensor = torch.stack([
            w[f"blocks.{x}.ffn.time_mix_r"] for x in range(n_layers)])
        self.key_ffn: torch.Tensor = torch.stack(
            [w[f"blocks.{x}.ffn.key.weight"] for x in range(n_layers)])
        self.receptance_ffn: torch.Tensor = torch.stack([
            w[f"blocks.{x}.ffn.receptance.weight"] for x in range(n_layers)])
        self.value_ffn: torch.Tensor = torch.stack([
            w[f"blocks.{x}.ffn.value.weight"] for x in range(n_layers)])

    def doLayer(self, x, statea, stateb, statec, stated, xx: int):

        xy = F.layer_norm(x, self.ln1w[xx], self.ln1b[xx])
        ct = torch.cat([torch.unsqueeze(statea, 0), xy[:-1]])

        kk = matvec(
            self.key[xx], torch.lerp(ct, xy, self.kktk[xx]))

        v = matvec(self.value[xx], torch.lerp(
            ct, xy, self.vvtv[xx]))
        rr = matvec(
            self.receptance[xx], torch.lerp(ct, xy, self.rrtr[xx]))
        r = torch.sigmoid(rr)
        k = torch.exp(kk)
        rz = []
        for i in range(lenn(x)):
            wrd =  (stateb + k[i] * v[i] * self.time_first[xx]) / (statec + k[i] * self.time_first[xx])

            stateb = self.time_decay[xx] * (stateb + k[i] * v[i])
            statec = (statec + k[i]) * self.time_decay[xx]

            rz += [wrd]
        mvv = x + matvec(self.outputvv[xx], r * torch.stack(rz))

        ddd = F.layer_norm(mvv, self.ln2w[xx], self.ln2b[xx])

        ctt = torch.cat([torch.unsqueeze(stated, 0), ddd[:-1]])

        km = torch.relu(matvec(self.key_ffn[xx], torch.lerp(
            ctt, ddd, self.time_mix_k_ffn[xx])))

        rt = torch.sigmoid((matvec(self.receptance_ffn[xx], torch.lerp(
            ctt, ddd, self.time_mix_r_ffn[xx]))))

        x = mvv + matvec(self.value_ffn[xx], km * km) * rt
        return x, xy[-1], stateb, statec, ddd[-1]

    def forward(self, token, state):
        x = F.layer_norm(self.emb[token], self.emb1, self.emb2)

        statea = state[0::4]
        stateb = state[1::4]
        statec = state[2::4]
        stated = state[3::4]

        ot = []

        for i in range(self.n_layers):
            x, aaa, bbb, ccc, ddd = self.doLayer(
                x, statea[i], stateb[i], statec[i], stated[i], i)

            ot = ot + [aaa, bbb, ccc, ddd]

        x = matvec(self.postprocess2, F.layer_norm(x, self.postprocess0,
                                                            self.postprocess1))   

        return x[-1], torch.stack(ot)
    
    