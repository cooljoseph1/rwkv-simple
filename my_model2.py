########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch
from torch.nn import functional as F
from tokenizers import Tokenizer

import torch.nn as nn

########################################################################################################

class RWKV_RNN(nn.Module):
    def __init__(self, model_path, n_layer=24, n_embed=1024):
        super().__init__()
        self.model_path = model_path
        self.n_layer = n_layer
        self.n_embed = n_embed

        self.eval() # set torch to inference mode
        
        """
        From here to the end of the function is really messy code to convert how the model
        weights were saved into an actual usable model.
        """
        w = torch.load(self.model_path, map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
            else: w[k] = w[k].float() # convert to f32 type
        
        
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

        blocks = []
        for i in range(self.n_layer):
            b = self.w.blocks[i]
            attention = b.att
            feed_forward_network = b.ffn
            layer_norm1 = b.ln1
            layer_norm2 = b.ln2
            block = Block(
                self.n_embed,
                attention,
                feed_forward_network,
                layer_norm1,
                layer_norm2,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.n_embed,), weight=w.weight, bias=w.bias)

    def forward(self, token, state):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.n_layer * 5, self.n_embed)
                for i in range(self.n_layer): state[5*i+4] = -1e30 # -infinity
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i, block in enumerate(self.blocks):
                x, state[5*i : 5*(i + 1)] = block.forward(x, state[5*i : 5*(i + 1)])


                
            
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state

##########################################################################################################

class Block(nn.Module):
    def __init__(self, n_embed, attention, feed_forward_network, layer_norm1, layer_norm2):
        super().__init__()
        self.n_embed = n_embed
        self.attention = AttentionLayer(
            attention.key.weight, attention.value.weight, attention.receptance.weight, attention.output.weight,
            attention.time_mix_k, attention.time_mix_v, attention.time_mix_r, attention.time_first, attention.time_decay, 
        )
        self.feed_forward_network = FeedForwardNetwork(
            feed_forward_network.key.weight,
            feed_forward_network.value.weight,
            feed_forward_network.receptance.weight,
            feed_forward_network.time_mix_k,
            feed_forward_network.time_mix_r,
        )
        self.layer_norm1 = layer_norm1
        self.layer_norm2 = layer_norm2

    def forward(self, x, state):
        x += self.attention.time_mixing(
            self.layer_norm(x, self.layer_norm1),
            state
        )
        x += self.feed_forward_network.channel_mixing(
            self.layer_norm(x, self.layer_norm2),
            state,
        )
        return x, state
        
    
    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.n_embed,), weight=w.weight, bias=w.bias)


########################################################################################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, key, value, receptance, time_mix_k, time_mix_r):
        super().__init__()

        self.key = key
        self.value = value
        self.receptance = receptance
        self.time_mix_k = time_mix_k
        self.time_mix_r = time_mix_r

    
    def channel_mixing(self, x, state):
        xk = x * self.time_mix_k + state[0] * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + state[0] * (1 - self.time_mix_r)
        state[0] = x
        r = torch.sigmoid(self.receptance @ xr)
        k = torch.square(torch.relu(self.key @ xk)) # square relu, primer paper
        return r * (self.value @ k)
    


########################################################################################################

class AttentionLayer(nn.Module):
    def __init__(self, key, value, receptance, output_weight,
            time_mix_k, time_mix_v, time_mix_r, time_first, time_decay):
        super().__init__()

        self.key = key
        self.value = value
        self.receptance = receptance
        self.output_weight = output_weight
        self.time_mix_k = time_mix_k
        self.time_mix_v = time_mix_v
        self.time_mix_r = time_mix_r
        self.time_first = time_first
        self.time_decay = time_decay


    def time_mixing(self, x, state):
        xk = x * self.time_mix_k + state[1] * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + state[1] * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + state[1] * (1 - self.time_mix_r)
        state[1] = x
        r = torch.sigmoid(self.receptance @ xr)
        k = self.key @ xk
        v = self.value @ xv
        
        aa = state[2]
        bb = state[3]
        pp = state[4]
        ww = self.time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + self.time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[2] = e1 * aa + e2 * v
        state[3] = e1 * bb + e2
        state[4] = qq
        return self.output_weight @ (r * wkv)
