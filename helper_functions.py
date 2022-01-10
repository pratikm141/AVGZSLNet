import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,Dataset
from torch import optim
import torch.utils.data as torchdata
import numpy as np
import scipy.io as sio
import math, random
import argparse, os
from sklearn.metrics import accuracy_score
import h5py, csv, pandas
import time
from sklearn.decomposition import PCA

class Features(nn.Module):
	def __init__(self, NetStruct, dropout_val):
		super(Features, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(len(NetStruct) - 1):
			self.layers.append(nn.Linear(NetStruct[i],NetStruct[i+1]))
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout_val)
	def forward(self, x):

		if len(self.layers) > 1:
			
			for i in range(len(self.layers)-1):
				x = self.layers[i](x)
				x = self.dropout(x)
				x = self.relu(x)
			y = self.layers[i+1](x)
		
		if len(self.layers) == 1:
			
			y = self.layers[0](x)

		return y  



class CombinedNetwork(nn.Module):
	def __init__(self,audio_net,video_net, text_net):
		super(CombinedNetwork, self).__init__()
		self.audio_net = audio_net
		self.video_net = video_net
		self.text_net = text_net

	def forward(self,input_audio, input_video, input_text):

		audio_out = self.audio_net(input_audio)
		video_out = self.video_net(input_video)
		text_out = self.text_net(input_text)
		return audio_out, video_out, text_out



def trn_tst_separate(all_cls_txt, trn_cls_txt, tst_cls_txt):

	with open(all_cls_txt, 'r') as fp:
		all_labels = [l.strip(",\n'") for l in fp]

	with open(trn_cls_txt, 'r') as fp:
		train_labels = [l.strip(",\n'") for l in fp]

	with open(tst_cls_txt, 'r') as fp:
		test_labels = [l.strip(",\n'") for l in fp]

	return all_labels, train_labels, test_labels


def make_dataset(root, mode, classes):
	root_dir = os.path.join(root, mode)
	dataAll = []
	labelAll = []

	print(classes)
	for i, target in enumerate(classes):
		d = os.path.join(root_dir, target+'.h5')
		hf = h5py.File(d, 'r')
		d = torch.from_numpy(hf['data'][()])
		hf.close()
		l = torch.tensor(classes.index(target))
		l = l.expand(d.shape[0])

		dataAll.append(d)
		labelAll.append(l)

	data, label = torch.cat(dataAll, dim=0), torch.cat(labelAll, dim=0)
	return data, label


class trnDataset_audio(Dataset):
	def __init__(self, root_data_audio,  labels_txt, mode ='trn'):
		
		self.audio_data_all, self.audio_label_all = make_dataset(root_data_audio, mode, labels_txt)
				
		self.labels_txt = labels_txt
		
			# Override to give PyTorch access to any image on the dataset
	def __getitem__(self, index):
		
		audio_labels, audio_data = self.audio_label_all, self.audio_data_all

		return audio_data[index],audio_labels[index]

	def __len__(self):
		return self.audio_data_all.shape[0]


class trnDataset_video(Dataset):
	def __init__(self, root_data_video,  labels_txt, mode ='trn'):
		
		
		self.video_data_all, self.video_label_all = make_dataset(root_data_video, mode, labels_txt)
				
		self.labels_txt = labels_txt
		
			# Override to give PyTorch access to any image on the dataset
	def __getitem__(self, index):
		
		video_labels, video_data = self.video_label_all, self.video_data_all
		return video_data[index],video_labels[index]

	def __len__(self):
		return self.video_data_all.shape[0]



class trnDatasetAV_avg_fix(Dataset):
	def __init__(self, root_data_audio, root_data_video, path_text_embeddings, root_path_data, zeroshot, labels_txt, mode ='trn'):
		
		self.audio_data_all, self.audio_label_all = make_dataset(root_data_audio, mode, labels_txt)
		self.video_data_all, self.video_label_all = make_dataset(root_data_video, mode, labels_txt)
				
		self.text_embeddings = np.load(path_text_embeddings,allow_pickle=True).item()
		self.labels_txt = labels_txt

		final_data_path = os.path.join(root_path_data, 'triplets_zeroshot_'+str(zeroshot)+'.csv')
		self.cls_slctd = pandas.read_csv(final_data_path,header=None)

		self.cls_cnt = [0 for ii in range(len(labels_txt))]
		
			# Override to give PyTorch access to any image on the dataset
	def __getitem__(self, index):

		audio_labels, audio_data = self.audio_label_all, self.audio_data_all
		video_labels, video_data = self.video_label_all, self.video_data_all
		
		pos_cls, neg_cls = self.cls_slctd.iloc[index].tolist()
		
		pos_text = torch.from_numpy(self.text_embeddings[pos_cls]) 
		neg_text = torch.from_numpy(self.text_embeddings[neg_cls])
		
		pos_index = np.asarray(np.where(audio_labels == self.labels_txt.index(pos_cls)))

		class_indx = self.labels_txt.index(pos_cls)

		l_cls=len(pos_index[0])
		if(self.cls_cnt[class_indx]>=l_cls-1):
			self.cls_cnt[class_indx] = 0
		else:
			self.cls_cnt[class_indx] = self.cls_cnt[class_indx] + 1
		pos_sample_index = pos_index[0][self.cls_cnt[class_indx]]

		neg_index = np.asarray(np.where(audio_labels == self.labels_txt.index(neg_cls)))
		neg_sample_index = np.random.choice(neg_index[0])
		
		
		pos_audio_data, neg_audio_data  = audio_data[pos_sample_index], audio_data[neg_sample_index]
		pos_video_data, neg_video_data = video_data[pos_sample_index], video_data[neg_sample_index]
		
		return pos_audio_data, neg_audio_data, pos_video_data, neg_video_data, pos_text, neg_text, self.labels_txt.index(pos_cls), self.labels_txt.index(neg_cls)
		

	def __len__(self):
		return self.cls_slctd.shape[0]



