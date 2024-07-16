import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_sparse import SparseTensor, spmm
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
#引入karate数据集以及cora
from torch_geometric.datasets import KarateClub, Planetoid

test_tensor = torch.tensor([[1., 2., 3.], 
                            [3., 2., 1.],
                            [1., 1., 1.]])
print(torch_scatter.scatter(test_tensor, index=torch.tensor([0, 0,1]), dim=1,dim_size=3, reduce='sum'))

def aggregate(inputs, index, dim_size = None):

        out = None

        if dim_size is None:
            dim_size = index.max().item() + 1
        out_features_dim = inputs.size(1)  # 特征维度
        out = torch.zeros((dim_size, out_features_dim), device=inputs.device)

        # 遍历所有节点
        for node_id in range(dim_size):
            # 找到当前节点的所有邻居节点索引
            neighbors = (index == node_id).nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                # 提取邻居节点的特征并计算平均值
                neighbor_features = inputs[neighbors]
                out[node_id] = neighbor_features.mean(dim=0)


        return out
    

    
class BuildGcn(nn.Module):
    #严格按照gcn那个d-1/2*A*D-1/2*X*W的公式来写
    def __init__(self,in_channels,out_channels):
        super(BuildGcn,self).__init__()
        self.lin=Linear(in_channels,out_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self,x,edge_index):
        edge_index_with_self_loops, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = pyg_utils.degree(edge_index_with_self_loops[1], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)

        # 获取a_hat邻接矩阵
        row, col = edge_index_with_self_loops
        edge_weight = torch.ones((row.size(0),), dtype=x.dtype)
        a_hat = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(x.size(0), x.size(0)))

        # 完成D^-1/2 * A * D^-1/2 * X * W
        step1 = deg_inv_sqrt_matrix @ a_hat.to_dense() @ deg_inv_sqrt_matrix
        x = step1 @ x
        x = self.lin(x)
        return x

class MyGraphSage(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        #如同 BuildGcn 一样，严格按照论文里的公式来进行复现
        super(MyGraphSage, self).__init__()
        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels)
    def forward(self, x, edge_index):
        #先进行聚合操作
        out = self.aggregate(x, edge_index)
        #再进行线性变换
        #out = F.relu(self.lin_l(torch.cat([x, out], dim=1)))
        out = F.relu(self.lin_l(x)+self.lin_r(out))
        norm = out.norm(p=2, dim=1, keepdim=True) + 1e-6  # 计算L2范数并避免除以0
        out = out / norm  # 归一化
        return out

    def aggregate(self,x,edge_index):#使用的是均值聚合
        #不要用scatter，用for循环好理解
        out=torch.zeros(x.size(0),x.size(1))
        for i in range(x.size(0)):
            neighbors=edge_index[1][edge_index[0]==i]
            if len(neighbors)>0:
                out[i]=x[neighbors].mean(dim=0)
        
        return out


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = BuildGcn(cora_dataset.num_features,32)
        self.conv2 = MyGraphSage(32, 16)
        self.lin = Linear(16, 7)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.lin(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.softmax(x, dim=1)
        return x
    
class SGC(torch.nn.Module):#复现Simple Graph Convolution
    def __init__(self,in_channels,out_channels,num_of_norm=2):
        super(SGC,self).__init__()
        self.lin = Linear(in_channels, out_channels)
        self.num_of_norm=num_of_norm
        
    def forward(self,x,edge_index):
        edge_index_with_self_loops, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = pyg_utils.degree(edge_index_with_self_loops[1], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)

        # 获取a_hat邻接矩阵
        row, col = edge_index_with_self_loops
        edge_weight = torch.ones((row.size(0),), dtype=x.dtype)
        a_hat = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(x.size(0), x.size(0)))

        # 完成D^-1/2 * A * D^-1/2 * X * W
        norm_adj_matrix = deg_inv_sqrt_matrix @ a_hat.to_dense() @ deg_inv_sqrt_matrix
        for i in range(self.num_of_norm):
            norm_adj_matrix = norm_adj_matrix @ norm_adj_matrix
        x = norm_adj_matrix @ x
        return F.softmax(self.lin(x),dim=1)
    




cora_dataset = Planetoid(root='cora', name='Cora')

train_loader = DataLoader(cora_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(cora_dataset, batch_size=32, shuffle=False)
model=SGC(cora_dataset.num_features,cora_dataset.num_classes,3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 注意：添加data.batch以支持批处理
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    total=0
    for data in loader:
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
        total += data.test_mask.sum().item() 
    return correct / total


best_loss = float('inf')
patience = 10  # 设置耐心阈值
patience_counter = 0  # 初始化耐心计数器
for epoch in range(100):
    model.train()
    loss = train()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss))
    # 早停逻辑
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0  # 重置耐心计数器
    else:
        patience_counter += 1  # 增加耐心计数器
    
    if patience_counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break  # 达到耐心阈值，停止训练
    
model.eval()
test_acc = test(test_loader)
print('Test Accuracy: {:.4f}'.format(test_acc))#第一次我自己改的aggressgate是0.7670，第二次leakyrelu直接0.7410，

#为何我自己写的矩阵乘法版本的gcn在cora数据集上Test Accuracy: 0.7140，论文里是0.81