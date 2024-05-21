# -*- codeing = utf-8 -*-
# @Time : 2024-05-20 17:03
# @Author : 张庭恺
# @File : model.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import Data
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
	embed_dim: int = 64
	head_num: int = 4
	dropout: float = 0.1
	vocab_size: int = 100070
	content_len: int = 16
	num_layers: int = 6


class Embadding(nn.Module):
	def __init__(self, vocab_size=100070, embed_dim=64):
		super(Embadding, self).__init__()
		self.input_embedding = nn.Embedding(vocab_size, embed_dim)

	def forward(self, x):
		return self.input_embedding(x)


# 余弦位子编码
class Sin_PositionEmbedding(nn.Module):
	def __init__(self, content_len=16, embed_dim=64):
		super(Sin_PositionEmbedding, self).__init__()
		# [16,64]
		self.position_embedding = torch.zeros(content_len, embed_dim).to('cuda')
		self.pre_dot = torch.arange(content_len)[:, None]
		self.Suffix_dot = (1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim)))
		self.position_embedding[:, 0::2] = torch.sin(self.pre_dot * self.Suffix_dot)
		self.position_embedding[:, 1::2] = torch.cos(torch.arange(content_len)[:, None] * (
				1.0 / (10000 ** (2 * (torch.arange(0, embed_dim, 2)).float() / embed_dim))))

	def forward(self, x):
		# [b,n,d] + [1,n,d]
		return x + self.position_embedding.unsqueeze(0)


class Causal_Atten(nn.Module):
	def __init__(self, embed_dim=64, head_num=8, dropout=0.1):
		super(Causal_Atten, self).__init__()
		self.q = nn.Linear(embed_dim, embed_dim)
		self.k = nn.Linear(embed_dim, embed_dim)
		self.v = nn.Linear(embed_dim, embed_dim)
		self.head_num = head_num
		self.dropout = nn.Dropout(dropout)
		self.proj = nn.Linear(embed_dim, embed_dim)

	def forward(self, x):
		b, n, d = x.size()
		'''
		改变形状
		'''
		q = self.q(x).view(b, n, self.head_num, -1).transpose(1, 2)  # [b,n,n_head,d/n_head] --> [b,n_head,n,d/n_head]
		k = self.k(x).view(b, n, self.head_num, -1).transpose(1, 2)  # [b,n,n_head,d/n_head] --> [b,n_head,n,d/n_head]
		v = self.v(x).view(b, n, self.head_num, -1).transpose(1, 2)  # [b,n,n_head,d/n_head] --> [b,n_head,n,d/n_head]
		attention = (q @ k.permute(0, 1, 3, 2)) * (torch.sqrt(torch.tensor(q.size(-1))))  # [b,n_head,n,n]

		# 添加掩码
		mask = torch.triu(torch.ones(n, n).view(1, 1, n, n), diagonal=1).to('cuda')
		attention = attention.masked_fill(mask == 1, float('-inf'))
		attention = F.softmax(attention, dim=-1)

		attention = self.dropout(attention)
		out = (attention @ v).transpose(1, 2).contiguous().view(b, n,-1)  # [b, n_head, n, n] @ [b, n_head, n, d/n_head] --> [b, n_head, n, d/n_head]
		return self.proj(out)


class MLP(nn.Module):
	def __init__(self, embed_dim=64, hidden_dim=256, dropout=0.1):
		super(MLP, self).__init__()
		self.l1 = nn.Linear(embed_dim, hidden_dim)
		self.act = nn.GELU()
		self.l2 = nn.Linear(hidden_dim, embed_dim)
		self.drop = nn.Dropout(dropout)

	def forward(self, x):
		x = self.l1(x)
		x = self.act(x)
		x = self.l2(x)
		x = self.drop(x)
		return x


class LayerNorm(nn.Module):
	def __init__(self, embed_dim, eps=1e-5):
		super(LayerNorm, self).__init__()
		self.layerNoem = nn.LayerNorm(embed_dim, eps)

	def forward(self, x):
		return self.layerNoem(x)


class GPT_Block(nn.Module):
	def __init__(self, embed_dim, num_head, hidden_dim, dropout):
		super(GPT_Block, self).__init__()
		self.norm1 = LayerNorm(embed_dim)
		self.atten = Causal_Atten(embed_dim, num_head, dropout)
		self.norm2 = LayerNorm(embed_dim)
		self.mlp = MLP(embed_dim, hidden_dim, dropout)

	def forward(self, x):
		res1 = x
		x = self.norm1(self.atten(x)) + res1
		res2 = x
		x = self.norm2(self.mlp(x)) + res2
		return x


class GPT(nn.Module):
	def __init__(self, vocab_size=100070, embed_dim=64, hidden_dim=256, num_layer=6, num_head=8, dropout=0.1,
	             content_len=16):
		super(GPT, self).__init__()
		self.embed = Embadding(vocab_size, embed_dim)
		self.pos_embed = Sin_PositionEmbedding(content_len=content_len, embed_dim=embed_dim)
		self.blocks = nn.ModuleList([GPT_Block(embed_dim, num_head,hidden_dim, dropout) for _ in range(num_layer)])

		self.proj = nn.Linear(embed_dim, vocab_size)
		self.apply(self._init_weights)
	def get_numel(self):
		return sum([param.numel() for param in self.parameters()])

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
	def forward(self, x,target = None):     # x[batch , content_len]
		embed = self.embed(x)               # [batch , content_len, embed_dim]
		pos_embed = self.pos_embed(embed)   # [batch , content_len, embed_dim]
		out = pos_embed
		for mod in self.blocks:
			out = mod(out)

		out = self.proj(out)                # [batch , content_len, vocab_size]
		if target is not None:
			logits = F.softmax(out,-1)
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),ignore_index=-1,reduction='mean')
			return loss
		else:
			logits = F.softmax(out,-1)
			out = logits[:,[-1],:]
			return out



