#做Do Transformer Really Perform Bad on Graph Reprensentation?的复现
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader,Data
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as utils
from torch_geometric.utils import to_networkx
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_sparse import SparseTensor, spmm
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax,to_dense_adj,degree
from torch_geometric.nn import GATConv,GCNConv,GraphSAGE
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub, Planetoid
from sklearn.manifold import TSNE
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy import sparse as sp

class SGFormer(nn.Module):
    def __init__(self, in_channels, out_channels,beta=0.4,alpha=0.6):
        super(SGFormer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.beta = beta
        self.alpha = alpha
        self.Wq = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.Wk = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.Wv = nn.Parameter(torch.Tensor(in_channels, in_channels))#这里的V的维度和q,k不一样都得是in
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = GCNConv(in_channels, in_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.Wq)
        nn.init.xavier_normal_(self.Wk)
        nn.init.xavier_normal_(self.Wv)
        

    def forward(self,x,edge_index):
        num_nodes = x.shape[0]
        self.one_column_vector = nn.Parameter(torch.ones(num_nodes,1)).to(x.device)
        q = torch.matmul(x,self.Wq)#(N, out_channels)
        q_hat = q/torch.norm(q,p='fro')
        k = torch.matmul(x,self.Wk)
        k_hat = k/torch.norm(k,p='fro')
        v = torch.matmul(x,self.Wv)#[N, out_channels]
        qk = (q_hat@(k_hat.t()@self.one_column_vector))/self.in_channels
        # print(qk.shape)
        # print((self.one_column_vector+qk).shape)
        #print(((self.one_column_vector+qk).squeeze(1)).shape)
        D = torch.diag((self.one_column_vector+qk).squeeze(1))#这里得把加起来后得到的向量给squeeze掉
        #print(D.shape)
        #diag([N,1]+[N,out]*([out,N]@[N,1]))=[N,N]
        Z = self.beta*torch.inverse(D)@(v+(q_hat@(k_hat.t()@v))/self.in_channels)+(1-self.beta)*x
        #[N,N]@([N,in]+[N,out]@([out,N]@[N,in]))+[N,in] = [N,in]+[N,in]=[N,in]
        gcn = self.conv1(x,edge_index)
        self.dropout(gcn)
        Z_out = (1-self.alpha)*Z+self.alpha*gcn
        #[N,in]+[N,in] = [N,in]
        Z_out = Z_out.to(x.device)
        return self.linear(Z_out)
    
def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        tsne = TSNE(n_components=2)
        h_2d = tsne.fit_transform(h)
        plt.scatter(h_2d[:, 0], h_2d[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()

cora_dataset = Planetoid(root='cora', name='Cora')
citeseer_dataset = Planetoid(root='citeseer', name='CiteSeer')
pubmed_dataset = Planetoid(root='pubmed', name='PubMed')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
used_dataset = cora_dataset
print('now my device is:',device)
train_loader = DataLoader(used_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(used_dataset, batch_size=64, shuffle=False)
model=SGFormer(used_dataset.num_features,used_dataset.num_classes).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 注意：添加data.batch以支持批处理
        loss = criterion(out[data.train_mask], data.y[data.train_mask]).to(device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1).to(device)
        correct += int((pred[data.train_mask] == data.y[data.train_mask]).sum())
        total = data.train_mask.sum().item()
    train_acc = correct / total
    return total_loss / len(train_loader),train_acc

def test(loader):
    model.eval()
    correct = 0
    total=0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index).to(device)
        pred = out.argmax(dim=1).to(device)
        correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
        total += data.test_mask.sum().item() 
    return correct / total
for epoch in range(200):
    model.train()
    loss,train_acc = train()
    print('Epoch: {}, Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, loss, train_acc))
    
model.eval()
test_acc = test(test_loader)
h = model(used_dataset[0].x.to(device), used_dataset[0].edge_index.to(device))
visualize(h, color=used_dataset[0].y)
print('Test Accuracy: {:.4f}'.format(test_acc))