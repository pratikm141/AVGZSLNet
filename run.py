import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader,Dataset
from torch import optim
import torch.utils.data as torchdata
import numpy as np
import math, random
import argparse, os, sys
import h5py, csv, pandas
import time
from helper_functions import *
torch.multiprocessing.set_sharing_strategy('file_system')
host_name = socket.gethostname()


def triplet_loss(embed1, data1, data2, margin, red = 'mean'):
    criterion_triplet = nn.TripletMarginLoss(margin, p=2, reduction=red)
    loss = criterion_triplet(embed1, data1, data2)
    return loss

def composite_triplet_loss(embed1, data1, embed2, data2, margin, red = 'mean'):
    criterion_triplet = nn.TripletMarginLoss(margin, p=2, reduction=red)
    loss1 = criterion_triplet(embed1, data1, data2) + criterion_triplet(embed2, data2, data1)
    loss2 = criterion_triplet(data1, embed1, embed2) + criterion_triplet(data2, embed2, embed1)
    loss = loss2+loss1
    return loss
parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default = 0.3, help='learning rate')
parser.add_argument('-batch_size', type=int, default = 4096, help='Batch Size')
parser.add_argument('-nepoch', type=int, default = 200, help='No. of epochs')
parser.add_argument('-optimizer', type=str, default = 'sgd', help='sgd or adam')
parser.add_argument('-margin', type=float, default = 1.0, help='Margin for tripplet loss')
parser.add_argument('-normalize', type=str, default = 'n', help='Whether to normalize input or not: y/n')
parser.add_argument('-audio_net', nargs='+', type=int, default=[1024, 256, 64])
parser.add_argument('-video_net', nargs='+', type=int, default=[1024, 256, 64])
parser.add_argument('-text_net', nargs='+', type=int, default=[300, 64])
parser.add_argument('-schedule', type=str, default = 'y', help='Whether to use learning rate scheduler')
parser.add_argument('-gpu', type=str, default = '2,3', help = 'which gpu to use' )
parser.add_argument('-dropout', type=float, default = 0.5, help='dropout for the network')
parser.add_argument('-version', type=str, default='1', help='co-eff for triplet loss')	
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(args.gpu)

def save_checkpoint(state, file_path):
	torch.save(state, file_path)


# parameters for training 
batch_size, nepoch, learning_rate = args.batch_size, args.nepoch, args.lr 
margin, optimizer_criteria = args.margin, args.optimizer

normalize, schedule = False, False
zeroshot = True

if args.normalize == 'y':
	normalize = True


if args.schedule == 'y':
	schedule = True

NetworkStructure_audio = args.audio_net
net_audio = Features(NetworkStructure_audio, args.dropout).cuda()

NetworkStructure_video = args.video_net
net_video = Features(NetworkStructure_video, args.dropout).cuda()

NetworkStructure_text = args.text_net
net_text = Features(NetworkStructure_text, 0.0).cuda()

net = CombinedNetwork(net_audio, net_video, net_text)

net.dec = nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,300))
net.cls = nn.Sequential(nn.Linear(64,48),nn.ReLU(),nn.Linear(48,23))

net = torch.nn.DataParallel(net)
net =net.cuda()

root_inp_audio_data = '../../smallAudioset/audio/'
root_inp_video_data = '../../smallAudioset/video/'
path_text_embedding = '../../smallAudioset/text/word_embeddings-dict.npy'
root_path_triplets = '../dataset_meta-files'
root_out = './output/cross-modal/avgzsl'+'_v'+str(args.version)
model_out_root = './models/cross-modal/avgzsl'+'_v'+str(args.version)

if not(os.path.isdir(os.path.join(root_out, 'layer-'+str(len(NetworkStructure_audio)-2)))):
	os.makedirs(os.path.join(root_out, 'layer-'+str(len(NetworkStructure_audio)-2)),exist_ok=True)
folder_out = 'lr-'+str(learning_rate)+'_batch_size-'+str(batch_size)+'_Nepoch-'+str(nepoch)+'_margin-'+str(margin)+'_zerosht-' + \
    '_normalize-'+str(normalize) +'_scheduler-' + \
    str(schedule)+'_dropout-'+str(args.dropout)

out_path = os.path.join(root_out, 'layer-'+str(len(NetworkStructure_audio)-2),folder_out)

os.makedirs(out_path,exist_ok=True)


if not(os.path.isdir(os.path.join(model_out_root, 'layer-'+str(len(NetworkStructure_audio)-2)))):
	os.makedirs(os.path.join(model_out_root, 'layer-'+str(len(NetworkStructure_audio)-2)),exist_ok=True)
model_path = os.path.join(model_out_root, 'layer-'+str(len(NetworkStructure_audio)-2), folder_out)

os.makedirs(model_path,exist_ok=True)

all_cls_txt = os.path.join(root_path_triplets, 'all_class.txt')
trn_cls_txt = os.path.join(root_path_triplets, 'train_class.txt')
tst_cls_txt = os.path.join(root_path_triplets, 'test_class.txt')
all_labels, train_labels, test_labels = trn_tst_separate(all_cls_txt, trn_cls_txt, tst_cls_txt)
seen_class = [all_labels.index(lbl) for lbl in train_labels]
unseen_class = [all_labels.index(lbl) for lbl in test_labels]
testing_space = {'u': test_labels, 's': train_labels, 't': all_labels}



labels = train_labels


labels = all_labels

dset_train = trnDatasetAV_avg_fix(root_inp_audio_data, root_inp_video_data, path_text_embedding, root_path_triplets, zeroshot, labels)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle =False, num_workers=4)


dset_audio = trnDataset_audio(root_inp_audio_data,  labels)
dset_audio_loader = DataLoader(dset_audio, batch_size=batch_size, shuffle =True, num_workers=4)


dset_video = trnDataset_video(root_inp_video_data,  labels)
dset_video_loader = DataLoader(dset_video, batch_size=batch_size, shuffle =True, num_workers=4)

criterion_mse = nn.MSELoss(reduce=True)

if optimizer_criteria == 'sgd':
	optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.9)
if optimizer_criteria == 'adam':
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

if schedule:
	scheduler = MultiStepLR(optimizer, milestones=[75, 125, 175], gamma=0.1,last_epoch=-1)


import tqdm
init=time.time()

import pdb
pdb.set_trace()
for epoch in tqdm.tqdm(range(nepoch)):
	# calculate audio average

	if schedule and epoch>0:
		scheduler.step()
	tot_loss, tot_loss_audio_triplet, tot_loss_video_triplet, tot_loss_triplet, tot_loss_crossmodal = 0.0, 0.0, 0.0, 0.0, 0.0
	tot_loss_avg = 0.0
	steps = 0

	for itr,data in enumerate(train_loader,0):
		steps+=1
		pos_audio, neg_audio, pos_video, neg_video, pos_text_orig, neg_text_orig, pos_cls, neg_cls = data

		

		if normalize:
			pos_audio, neg_audio = F.normalize(pos_audio, p=2, dim=1), F.normalize(neg_audio, p=2, dim=1)
			pos_video, neg_video = F.normalize(pos_video, p=2, dim=1), F.normalize(neg_video, p=2, dim=1)
			pos_text_orig, neg_text_orig = F.normalize(pos_text_orig, p=2, dim=1), F.normalize(neg_text_orig, p=2, dim=1)

		
		pos_audio, neg_audio = Variable(pos_audio.cuda().float()), Variable(neg_audio.cuda().float())
		pos_video, neg_video = Variable(pos_video.cuda().float()), Variable(neg_video.cuda().float())
		pos_text_orig, neg_text_orig = Variable(pos_text_orig.cuda().float()), Variable(neg_text_orig.cuda().float())
		
		optimizer.zero_grad()
		
		pos_audio, pos_video, pos_text = net(pos_audio, pos_video, pos_text_orig)
		neg_audio, neg_video, neg_text = net(neg_audio, neg_video, neg_text_orig)

		#################################################################################
		
		# Loss Calculation
		loss_audio_triplet = composite_triplet_loss(pos_text, pos_audio, neg_text, neg_audio, margin)
		loss_video_triplet = composite_triplet_loss(pos_text, pos_video, neg_text, neg_video, margin)
		


		loss_composite_triplet = loss_audio_triplet +  loss_video_triplet


		pos_text_text = net.module.dec(pos_text)

		pos_audio_text = net.module.dec(pos_audio)
		neg_audio_text = net.module.dec(neg_audio)
		pos_video_text = net.module.dec(pos_video)
		neg_video_text = net.module.dec(neg_video)


		loss_cmd = criterion_mse(pos_text_text,pos_text_orig) + criterion_mse(pos_audio_text,pos_text_orig) + criterion_mse(pos_video_text,pos_text_orig) + triplet_loss(pos_text_text, pos_audio_text, neg_audio_text, margin) + triplet_loss(pos_text_text, pos_video_text, neg_video_text, margin) 

		loss =loss_composite_triplet + loss_cmd
		####################################################################################
		tot_loss += loss.item()
		loss.backward()
		optimizer.step()

		if itr %10 == 0 :
			print("Epoch number:{}, Iteration:{}, loss:{}, Time:{:.2f}".format(epoch,itr,tot_loss/steps, time.time()-init))


			tot_loss = 0.0
			steps = 0
			init = time.time()

	if(epoch%10==0):
		filename = 'ep{}.pth.tar'.format(epoch)
		print('Saving Model')
		save_checkpoint({'epoch': epoch + 1,'state_dict': net.module.state_dict(),'optimizer': optimizer.state_dict(),}, os.path.join(model_path, filename))

filename = 'final_model.pth.tar'
print('Saving Model')
save_checkpoint({'epoch': epoch + 1,'state_dict': net.module.state_dict(),'optimizer': optimizer.state_dict(),}, os.path.join(model_path, filename))
