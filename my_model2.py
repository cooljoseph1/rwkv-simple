########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, torch
from torch.nn import functional as F

import torch.nn as nn

########################################################################################################

class RWKV_RNN(nn.Module):
    def __init__(self, embeddings, layer_norm0, blocks, head, layer_norm_out, n_layer=24, n_embed=1024):
        super().__init__()
        self.n_layer = n_layer
        self.n_embed = n_embed

        self.embeddings = embeddings
        self.layer_norm0 = layer_norm0
        self.blocks = nn.ModuleList(blocks)
        self.head = head
        self.layer_norm_out = layer_norm_out

    @classmethod
    def from_blink_file(cls, path_to_file):
        """
        Given a file with a pretrained model from BlinkDL, convert this into
        a network.
        """
        ## Step 1: Load the weights from the file & modify them appropriately
        w = torch.load(path_to_file, map_location='cpu')
        for k in w.keys():
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_decay' in k: w[k] = -torch.exp(w[k].float()) # the real time decay is like e^{-e^x}
            else: w[k] = w[k].float()

        
        ## Step 2: Convert this into a namespace, putting the blocks in a dictionary
        namespace = types.SimpleNamespace()
        namespace.blocks = {}
        n_layer = 0
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = namespace
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    n_layer = max(n_layer, p + 1)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])
        
        ## Step 3: Build the different components of the network
        embeddings = namespace.emb.weight
        n_embed = len(embeddings[0])


        layer_norm0 = LayerNorm(
            n_embed,
            namespace.blocks[0].ln0.weight,
            namespace.blocks[0].ln0.bias,
        )

        blocks = []
        for i in range(n_layer):
            b = namespace.blocks[i]
            attention = AttentionLayer.from_namespace(b.att)
            feed_forward_network = FeedForwardNetwork.from_namespace(b.ffn)
            layer_norm1 = LayerNorm(n_embed, b.ln1.weight, b.ln1.bias)
            layer_norm2 = LayerNorm(n_embed, b.ln2.weight, b.ln2.bias)
            block = Block(
                n_embed,
                attention,
                layer_norm1,
                feed_forward_network,
                layer_norm2,
            )
            blocks.append(block)

        head = namespace.head.weight
        layer_norm_out = LayerNorm(n_embed, namespace.ln_out.weight, namespace.ln_out.bias)

        ## Finally, put all these components together in the main network
        return cls(embeddings, layer_norm0, blocks, head, layer_norm_out, n_layer, n_embed)
        

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.n_embed,), weight=w.weight, bias=w.bias)

    def forward(self, token, state):
        if state == None:
            state = torch.zeros(self.n_layer * 5, self.n_embed)
            state[4::5] = -1e30 # -infinity
        
        x = self.embeddings[token]
        x = self.layer_norm0.forward(x)
        for i, block in enumerate(self.blocks):
            x, state[5*i : 5*(i + 1)] = block.forward(x, state[5*i : 5*(i + 1)])
        
        x = self.layer_norm_out(x)
        x = self.head @ x
        return x.float(), state

##########################################################################################################

class LayerNorm(nn.Module):
    def __init__(self, n_embed, weight, bias):
        super().__init__()

        self.n_embed = n_embed
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return F.layer_norm(x, (self.n_embed,), weight=self.weight, bias=self.bias)

##########################################################################################################

class Block(nn.Module):
    """
    The main "transformer" blocks that make up the network. I put "transformer" in quotes because it's not
    really a transformer (since it doesn't use the same kind of attention mechanism).
    """
    def __init__(self, n_embed, attention, layer_norm1, feed_forward_network, layer_norm2):
        super().__init__()
        self.n_embed = n_embed
        self.attention = attention
        self.layer_norm1 = layer_norm1
        self.feed_forward_network = feed_forward_network
        self.layer_norm2 = layer_norm2

    def forward(self, x, state):
        x += self.attention.time_mixing(
            self.layer_norm1.forward(x),
            state
        )
        x += self.feed_forward_network.channel_mixing(
            self.layer_norm2.forward(x),
            state,
        )
        return x, state
        


########################################################################################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, key, value, receptance, time_mix_k, time_mix_r):
        super().__init__()

        self.key = key
        self.value = value
        self.receptance = receptance
        self.time_mix_k = time_mix_k
        self.time_mix_r = time_mix_r

    @classmethod
    def from_namespace(cls, namespace):
        """
        Return a feed forward network layer given a namespace `ffn` that defines
        - ffn.key.weight,
        - ffn.value.weight,
        - ffn.receptance.weight,
        - ffn.time_mix_k,
        - ffn.time_mix_r,
        """
        return cls(
            namespace.key.weight,
            namespace.value.weight,
            namespace.receptance.weight,
            namespace.time_mix_k,
            namespace.time_mix_r,
        )

    
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

    @classmethod
    def from_namespace(cls, namespace):
        """
        Return an attention layer given a namespace `attention` that defines
        - attention.key.weight
        - attention.value.weight
        - attention.receptance.weight
        - attention.output.weight
        - attention.time_mix_k
        - attention.time_mix_v
        - attention.time_mix_r
        - attention.time_first
        - attention.time_decay
        """
        return cls(
            namespace.key.weight, namespace.value.weight, namespace.receptance.weight, namespace.output.weight,
            namespace.time_mix_k, namespace.time_mix_v, namespace.time_mix_r, namespace.time_first, namespace.time_decay, 
        )


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
