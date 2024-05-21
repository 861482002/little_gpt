# -*- codeing = utf-8 -*-
# @Time : 2024-05-21 10:28
# @Author : 张庭恺
# @File : main.py
# @Software : PyCharm
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from tiktoken import get_encoding
import argparse

from model import GPT
from tokenizer import Data


@dataclass
class GPT_Config:
	embed_dim: int = 64
	head_num: int = 4
	dropout: float = 0.1
	vocab_size: int = 100070
	content_len: int = 16
	hidden_dim: int = 256
	num_layers: int = 6
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	batch_size: int = 16
	lr: float = 1e-3
	epoch: int = 50

'''
1、构建数据集
2、构建模型
3、构建优化器
4、训练模型
'''

def get_batch(data, train: bool = True, cfg: GPT_Config = None):
	if train:
		train = data.train_data
		idxx = np.random.randint(0, len(train) - cfg.content_len, size=cfg.batch_size)
		x_batch = np.array([train[idx:idx + cfg.content_len] for idx in idxx])
		y_batch = np.array([train[idx + 1:idx + cfg.content_len + 1] for idx in idxx])
	else:
		val = data.val_data
		idxx = np.random.randint(0, len(val) - cfg.content_len, size=cfg.batch_size)
		x_batch = np.array([train[idx:idx + cfg.content_len] for idx in idxx])
		y_batch = np.array([train[idx + 1:idx + cfg.content_len + 1] for idx in idxx])
	return x_batch, y_batch


def main():
	cfg = GPT_Config()

	# 1、1数据集
	tokenizer = tiktoken.get_encoding("cl100k_base")
	data_path = './sales_textbook.txt'
	data = Data(data_path, tokenizer, cfg.content_len, cfg.batch_size)
	# 1、2 获取批量数据



	# 2、1构建模型
	model = GPT(cfg.vocab_size,cfg.embed_dim,cfg.hidden_dim,cfg.num_layers,cfg.head_num,cfg.dropout,cfg.content_len)
	param_numel = model.get_numel()
	print(f'模型参数量：{param_numel}')
	# 2、2将模型移动到GPU上
	model.to(cfg.device)
	# 3、构建优化器
	adam = optim.Adam(model.parameters(), lr=cfg.lr)

	# 4、训练模型
	for epoch in range(cfg.epoch):
		model.train()
		x_batch, y_batch = get_batch(data, train=True, cfg=cfg)
		# 将数据移动到GPU上
		x_batch,y_batch = torch.tensor(x_batch).to(cfg.device),torch.tensor(y_batch).type(torch.LongTensor).to(cfg.device)

		# 梯度清零
		adam.zero_grad()
		loss = model(x_batch,y_batch)
		loss.backward()
		adam.step()
		print(f'epoch{epoch + 1}  loss : {loss.item()}')
	print('训练完成')
if __name__ == '__main__':
	main()
