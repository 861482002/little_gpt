# -*- codeing = utf-8 -*-
# @Time : 2024-05-20 17:03
# @Author : 张庭恺
# @File : tokenizer.py
# @Software : PyCharm

import transformers

import requests
import os
import tiktoken
import numpy as np
from argparse import ArgumentParser

# 解析超参数
def get_args(arg_par:ArgumentParser):
	arg_par.add_argument('--context_length',type=int,default=16,help='the context length')
	arg_par.add_argument('--d_model',type=int,default=64,help='the dimension of model')
	arg_par.add_argument('--batch',type=int,default=4,help='the batch size')

	args = arg_par.parse_args()
	return args


def down_loaddata(path_or_url: str, save_path: str):
	if not os.path.exists(save_path):
		url = path_or_url
		response = requests.get(url).content
		with open(save_path, 'w') as f:
			f.write(response.decode('utf-8'))


def read_data(path: str):
	with open(path, 'r') as f:
		content = f.read()
	return content

def tokenize_data(encoder:tiktoken.core.Encoding,data):
	encoded_data = encoder.encode(data)
	return encoded_data

def split_train_val(data, ratio=0.8):
	train = int(len(data) * ratio)
	val = int(len(data) - train)
	train_data = data[:train]
	val_data = data[train:]
	return train_data, val_data

class Data:
	def __init__(self,data_path:str,tokenizer:tiktoken.core.Encoding,context_length:int,batch_size:int):
		data = self.read_data(data_path)
		self.encoded_data = tokenizer.encode(data)
		self._vocab_size = max(self.encoded_data) + 1
		self._train_data,self._val_data = self.split_train_val(self.encoded_data,ratio=0.8)

	@property
	def vocab_size(self):
		return self._vocab_size
	@property
	def train_data(self):
		return self._train_data

	@property
	def val_data(self):
		return self._val_data

	def read_data(self,path: str):
		with open(path, 'r') as f:
			content = f.read()
		return content

	def split_train_val(self,data, ratio=0.8):
		train = int(len(data) * ratio)
		val = int(len(data) - train)
		train_data = data[:train]
		val_data = data[train:]
		return train_data, val_data

if __name__ == '__main__':

	parser = ArgumentParser(description='this is a parser')
	args = get_args(parser)
	path = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
	down_loaddata(path, 'sales_textbook.txt')

	data = read_data('sales_textbook.txt')
	print(len(data))
	encoder = tiktoken.get_encoding("cl100k_base")
	tokened_data = tokenize_data(encoder, data)
	print(len(tokened_data))
	train,val = split_train_val(tokened_data)

	idxx = np.random.randint(0,len(train) - args.context_length,size=args.batch)
	x_batch = np.array([train[idx:idx+args.context_length] for idx in idxx])
	y_batch = np.array([train[idx+1:idx+args.context_length+1] for idx in idxx])
	print(x_batch.shape)
	print(encoder.decode(x_batch[1]))
	data = Data('sales_textbook.txt',encoder,args.context_length,args.batch)
	print(data.vocab_size)
	pass
