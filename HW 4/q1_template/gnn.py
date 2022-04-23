from pyexpat import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def get_dataset(save_path):
	'''
	read data from .npy file 
	no need to modify this function
	'''
	raw_data = np.load(save_path, allow_pickle=True)
	dataset = []
	for i, (node_f, edge_index, edge_attr, y)in enumerate(raw_data):
		sample = Data(
			x=torch.tensor(node_f, dtype=torch.float),
			y=torch.tensor([y], dtype=torch.float),
			edge_index=torch.tensor(edge_index, dtype=torch.long),
			edge_attr=torch.tensor(edge_attr, dtype=torch.float)
		)
		dataset.append(sample)
	return dataset


class GraphNet(nn.Module):
	'''
	Graph Neural Network class
	'''
	def __init__(self, n_features):
		'''
		n_features: number of features from dataset, should be 37
		'''
		super(GraphNet, self).__init__()
		# define your GNN model here
		self.conv1 = GCNConv(n_features,25)
		self.conv2 = GCNConv(25,18)
		self.conv3 = GCNConv(18,12)
		self.conv4 = GCNConv(12,1)
		#self.conv5 = GCNConv(5,1)
		#raise NotImplementedError
		
	def forward(self, data):
		# define the forward pass here
		x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
		x = F.gelu(self.conv1(x,edge_index))
		#x = F.dropout(x,training=self.training,p=0.1)
		x = F.relu(self.conv2(x,edge_index))
		#x = F.dropout(x,training=self.training,p=0.2)
		x = F.gelu(self.conv3(x,edge_index))
		#x = F.dropout(x,training=self.training,p=0.2)
		#x = F.relu(self.conv4(x,edge_index))
		x = self.conv4(x,edge_index)
		#x = F.log_softmax(x,dim=0)
		return scatter_mean(torch.squeeze(x),data.batch)
		#raise NotImplementedError
	

def main():
	# load data and build the data loader
	train_set = get_dataset('train_set.npy')
	test_set = get_dataset('test_set.npy')
	train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

	# number of features in the dataset
	# no need to change the value
	n_features = 37
	num_epoch = 300

	# build your GNN model
	model = GraphNet(n_features)
	#model.to(device)


	# define your loss and optimizer
	loss_func = nn.MSELoss()
	optimizer = optim.RMSprop(model.parameters(),lr=3e-3)
	#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.8,patience=10,cooldown=0,verbose=True)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = num_epoch,eta_min=1e-3)

	print(model)

	hist = {"train_loss":[], "test_loss":[]}
	
	for epoch in range(1, 1+num_epoch):
		model.train()
		loss_all = 0
		optimizer.zero_grad()
		for data in train_loader:
			# your codes for training the model
			# ...
			#data.to(device)
			out = model(data)
			loss = loss_func(out,data.y)
			loss.backward()
			optimizer.step()
			loss_all += loss.item() * data.num_graphs * len(data)
			
			
		scheduler.step(loss.item())
		train_loss = loss_all / len(train_set)

		with torch.no_grad():
			loss_all = 0
			for data in test_loader:
				# your codes for validation on test set
				# ...
				#data.to(device)
				out = model(data)
				loss = loss_func(out,data.y)
				loss_all += loss.item() * data.num_graphs * len(data)
			test_loss = loss_all / len(test_set)

			hist["train_loss"].append(train_loss)
			hist["test_loss"].append(test_loss)
			print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Test loss: {test_loss:.3}')
	# test on test set to get prediction 
	with torch.no_grad():
		prediction = np.zeros(len(test_set))
		label = np.zeros(len(test_set))
		idx = 0
		for data in test_loader:
			data = data.to(device)
			output = model(data)
			prediction[idx:idx+len(output)] = output.squeeze().detach().numpy()
			label[idx:idx+len(output)] = data.y.detach().numpy()
			idx += len(output)
		prediction = np.array(prediction).squeeze()
		label = np.array(label).squeeze()

	# visualization
	# plot loss function
	ax = plt.subplot(1,1,1)
	ax.plot([e for e in range(1,1+num_epoch)], hist["train_loss"], label="train loss")
	ax.plot([e for e in range(1,1+num_epoch)], hist["test_loss"], label="test loss")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	ax.legend()
	plt.show()

	# plot prediction vs. label
	x = np.linspace(np.min(label), np.max(label))
	y = np.linspace(np.min(label), np.max(label))
	ax = plt.subplot(1,1,1)
	ax.scatter(prediction, label, marker='+', c='red')
	ax.plot(x, y, '--')
	plt.xlabel("prediction")
	plt.ylabel("label")
	plt.show()

	print("MSE:", np.sum(np.square(prediction-label)))


if __name__ == "__main__":
	main()
