"""
This file is used to test the equivalence between my implementation of RWKV and rwkvstic's.
"""

import torch
from tqdm import tqdm

#############################################
#          Load the RwkvStic model          #
#############################################
# from rwkvstic.load import RWKV
# stic_model = RWKV(
#     "./model/RWKV-4b-Pile-436M-20230211-8012.pth",
#     mode="pytorch(cpu/gpu)",
#     useGPU=True,
#     runtimedtype=torch.bfloat16,
#     dtype=torch.bfloat16,
# )

##############################################
#        Load my version of the model        #
##############################################

model_path = './model/RWKV-4b-Pile-436M-20230211-8012.pth'
n_layer = 24
n_embd = 1024

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."

from my_model2 import RWKV_RNN

my_model = RWKV_RNN.from_blink_file(
    model_path,
)
# from load_model import RWKV
# my_model = RWKV(
#     "./model/RWKV-4b-Pile-436M-20230211-8012.pth",
# )


##############################################
#             Load the tokenizer             #
##############################################
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./model/20B_tokenizer.json")

##############################################
#               Test them out!               #
##############################################

# stic_model.loadContext(newctx=f"Q: who is Jim Butcher?\n\nA:")
# stic_output = stic_model.forward(number=1)["logits"]
# print(stic_output)


context = "\n\nQ: What is 2+2?\n\n"
init_state = None
for token in tokenizer.encode(context):
    init_out, init_state = my_model.forward(token, init_state)
print(init_out)