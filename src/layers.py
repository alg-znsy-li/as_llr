import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from deepspeed.utils import log_dist
from transformers import Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):

    def devide_matrix_by_svd(self, W, r):
        a = W.to(dtype=torch.float32)
        u, s, vh = torch.linalg.svd(a, full_matrices=False)
        v = vh.t()
        A = u[:, :r] @ torch.diag(s)[:r, :r] ** (0.5)
        B = torch.diag(s)[:r, :r] ** (0.5) @ (v[:, :r]).t()
        Wlora = A @ B
        Wres = u[:, r:] @ torch.diag(s)[r:, r:] @ (v[:, r:]).t()
        return A.to(dtype=torch.bfloat16), B.to(dtype=torch.bfloat16), \
            Wres.to(dtype=torch.bfloat16), Wlora.to(dtype=torch.bfloat16)

    def devide_matrix_by_svd_2(self, W, r_):
        a = W.to(dtype=torch.float32)
        u, s, vh = torch.linalg.svd(a, full_matrices=False)
        v = vh.t()
        r = W.shape[-1] - r_
        Wres = u[:, :r] @ torch.diag(s)[:r, :r]  @ (v[:, :r]).t()
        A = u[:, r:] @ (torch.diag(s)[r:, r:]**0.5)
        B = (torch.diag(s)[r:, r:]**0.5) @ (v[:, r:]).t()
        Wlora = A @ B
        return A.to(dtype=torch.bfloat16), B.to(dtype=torch.bfloat16), \
            Wres.to(dtype=torch.bfloat16), Wlora.to(dtype=torch.bfloat16)

    def __init__(self, linear, rank, alpha, pissa=False):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        if pissa:
            A, B, Wres, Wlora = self.devide_matrix_by_svd_2(self.linear.weight.data, rank)
            self.linear.weight.data.copy_(Wres)
            self.lora.A.data.copy_(A)
            self.lora.B.data.copy_(B)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def add_lora(model, r, alpha, pissa):
    print(model)
    print(model.model)
    print(model.model.layers)
    for layer in model.model.layers:
        """
        Qwen2DecoderLayer
        """
        layer.self_attn.q_proj = LinearWithLoRA(layer.self_attn.q_proj, r, alpha, pissa)
        layer.self_attn.k_proj = LinearWithLoRA(layer.self_attn.k_proj, r, alpha, pissa)
        layer.self_attn.v_proj = LinearWithLoRA(layer.self_attn.v_proj, r, alpha, pissa)
        layer.self_attn.o_proj = LinearWithLoRA(layer.self_attn.o_proj, r, alpha, pissa)
    log_dist(message="******************HERE")
    
def append_llama_pro_block_group(model, group_size):

    model_inner:Qwen2Model = model.model
    layers_new = nn.ModuleList([])
    layers_group = list()
    idx_new = 0
    config = model_inner.config
    for idx, layer in enumerate(model_inner.layers):
        ori_layer = Qwen2DecoderLayer(config, idx_new)
        for param in ori_layer.named_parameters():
            val = layer.state_dict()[param[0]]
            param[1].data = val.data.detach().clone()
            param[1].requires_grad = False

        idx_new += 1
        layers_group.append(ori_layer)

        if len(layers_group) % group_size == 0:
            for ele in layers_group:
                layers_new.append(ele)

            llama_pro_layer = Qwen2DecoderLayer(config, idx_new)
            for param in llama_pro_layer.named_parameters():
                log_dist(message=f"hhhhHHHHHHHHHHHHHHH&&&&&**************{param[0]}", ranks=[0])
                if param[0].startswith('mlp.down_proj') or param[0].startswith('self_attn.o_proj'): 
                    log_dist(message=f"**************{param[0]}", ranks=[0])
                    log_dist(message="\n\n\n\n\nHIT-----------llama_pro")
                    param[1].data = torch.zeros_like(input=llama_pro_layer.state_dict()[param[0]])
                else:
                    val = layers_group[-1].state_dict()[param[0]]
                    param[1].data = val.data.detach().clone()
                param[1].requires_grad = True

            idx_new += 1
            layers_new.append(llama_pro_layer)

            layers_group = list()

    if len(layers_group) > 0:
        for ele in layers_group:
            layers_new.append(ele)

        llama_pro_layer = Qwen2DecoderLayer(config, idx_new)
        for param in llama_pro_layer.named_parameters():
            if param[0].startswith('mlp.down_proj') or param[0].startswith('self_attn.o_proj'):
                log_dist(message="\n\n\n\n\nHIT-----------llama_pro")
                log_dist(message=f"**************{param[0]}", ranks=[0])
                param[1].data = torch.zeros_like(input=llama_pro_layer.state_dict()[param[0]])
            else:
                val = layers_group[-1].state_dict()[param[0]]
                param[1].data = val.data.detach().clone()
            param[1].requires_grad = True

        idx_new += 1
        layers_new.append(llama_pro_layer)

        layers_group = list()
    model_inner.layers = layers_new

