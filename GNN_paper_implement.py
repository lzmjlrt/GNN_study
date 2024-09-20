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
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax,to_dense_adj,degree
#引入GATConv
from torch_geometric.nn import GATConv
#引入karate数据集以及cora
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub, Planetoid
from sklearn.manifold import TSNE

# test_tensor = torch.tensor([[1., 2., 3.], 
#                             [3., 2., 1.],
#                             [1., 1., 1.]])
# print(torch_scatter.scatter(test_tensor, index=torch.tensor([0, 0,1]), dim=1,dim_size=3, reduce='sum'))


def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    edge_index = edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        adj[src, dst] = 1
        adj[dst, src] = 1  # 如果是无向图
    return adj


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
        device = x.device
        edge_index_with_self_loops, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = pyg_utils.degree(edge_index_with_self_loops[1], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)

        # 获取a_hat邻接矩阵
        row, col = edge_index_with_self_loops
        edge_weight = torch.ones((row.size(0),), dtype=x.dtype,device=device)
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
        out = self.aggregate(x, edge_index).to(x.device)
        #out = F.relu(self.lin_l(torch.cat([x, out], dim=1)))
        out = F.relu(self.lin_l(x)+self.lin_r(out))
        norm = out.norm(p=2, dim=1, keepdim=True) + 1e-6  # 计算L2范数并避免除以0
        out = out / norm  # 归一化
        return out

    def aggregate(self,x,edge_index):#使用的是均值聚合
        #不要用scatter，用for循环好理解
        out=torch.zeros(x.size(0),x.size(1),device=x.device)
        for i in range(x.size(0)):
            neighbors=edge_index[1][edge_index[0]==i]
            if len(neighbors)>0:
                out[i]=x[neighbors].mean(dim=0)
        
        return out
    
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.conv1 = nn.Linear(cora_dataset.num_node_features, 64)
        self.conv2 = nn.Linear(64, cora_dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x)
        x = F.dropout(F.relu(x), p=0.4, training=self.training)
        x = self.conv2(x)
        return x

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = BuildGcn(cora_dataset.num_features, 16)
        self.conv2 = BuildGcn(16, cora_dataset.num_classes)
        
            
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
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
        edge_weight = torch.ones((row.size(0),), dtype=x.dtype,device=x.device)
        a_hat = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(x.size(0), x.size(0)))

        # 完成D^-1/2 * A * D^-1/2 * X * W
        norm_adj_matrix = deg_inv_sqrt_matrix @ a_hat.to_dense() @ deg_inv_sqrt_matrix
        for i in range(self.num_of_norm):
            norm_adj_matrix = norm_adj_matrix @ norm_adj_matrix
        x = norm_adj_matrix @ x
        return F.softmax(self.lin(x),dim=1)
    
class NewSGC(MessagePassing):#利用消息传递的思想来复现SGC
    def __init__(self,in_channels,out_channels,K=2):
        super(NewSGC,self).__init__(aggr = 'mean')
        self.lin = Linear(in_channels, out_channels)
        self.K = K
        
    def forward(self,x,edge_index):
        edge_index,_ = add_self_loops(edge_index,num_nodes=x.size(0))
        row,col = edge_index
        deg=degree(row,x.size(0),dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        for _ in range(self.K):
           x = self.propagate(edge_index, x=x, norm=norm)
           
        x = self.lin(x)
        return F.softmax(x,dim=1)
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
    
    def update(self,aggr_out):
        return aggr_out
        
    
class GAT(torch.nn.Module):#复现GAT
    def __init__(self,in_channels,out_channels):
        super(GAT,self).__init__()
        self.lin = Linear(in_channels, out_channels)
        self.attention = Linear(2*out_channels,1)
    def forward(self,x,edge_index):
        device = x.device
        #加上self-loop
        edge_index_with_self_loops, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        #获得加上self-loop的邻接矩阵
        adj_matrix = to_dense_adj(edge_index_with_self_loops,max_num_nodes=x.size(0))[0].to(device)
        #构造一个和邻接矩阵一样大小的全0矩阵
        attention_matrix = torch.zeros_like(adj_matrix).to(device)
        #attention_matrix = torch.zeros((self.out_channels,x.size(0),x.size(0)),device=device)
        for i in range(x.size(0)):
            #获取他所有邻居节点的特征向量
            neighbors = adj_matrix[i].nonzero(as_tuple=False).squeeze().to(device)#neighbors是一个维度为n*1的tensor
            if len(neighbors)>0:
                denominator=0
                #使用for循环来迭代，不要用广播
                for j in neighbors:
                    neighbors_feature=x[j]
                    #whi_whk=torch.cat(self.lin(x[i].unsqueeze(0)),self.lin(neighbors_feature.unsqueeze(0)),dim=1)
                    whi_whk=torch.cat((self.lin(x[i].unsqueeze(0)), self.lin(neighbors_feature.unsqueeze(0))), dim=1)
                    denominator += torch.exp(F.leaky_relu(self.attention(whi_whk)))
                #下面就是处理分子部分，就是每个邻居节点的aj
                for j in neighbors:
                    neighbors_feature=x[j]
                    #whi_whj=torch.cat(self.lin(x[i].unsqueeze(0)),self.lin(neighbors_feature.unsqueeze(0)),dim=1)
                    whi_whj=torch.cat((self.lin(x[i].unsqueeze(0)), self.lin(neighbors_feature.unsqueeze(0))), dim=1)
                    ai_j=torch.exp(F.leaky_relu(self.attention(whi_whj)))/denominator
                    #print(ai_j)
                    #print(ai_j.shape)
                    attention_matrix[i,j]=ai_j.mean()
        out = F.elu(attention_matrix@self.lin(x))
        return out
    
class GraphAttentionLayer(torch.nn.Module):#这个是传统意义上的全局注意力，实际上效果很差
    def __init__(self,in_channels,out_channels,head=8):
        super(GraphAttentionLayer,self).__init__()
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.head=head
        self.Q =  nn.ParameterList([nn.Parameter(torch.Tensor(in_channels, out_channels)) for _ in range(head)])
        self.K =  nn.ParameterList([nn.Parameter(torch.Tensor(in_channels, out_channels)) for _ in range(head)])
        self.V =  nn.ParameterList([nn.Parameter(torch.Tensor(in_channels, in_channels)) for _ in range(head)])
        self.line=Linear(head*out_channels,out_channels)
        # self.linQ = Linear(in_channels, out_channels)
        # self.linK = Linear(in_channels, out_channels)
        # self.linV = Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        for q in self.Q:
            nn.init.xavier_normal_(q)
        for k in self.K:
            nn.init.xavier_normal_(k)
        for v in self.V:
            nn.init.xavier_normal_(v)
    
    def forward(self,x,edge_index):
        for head in range(self.head):
            self.Q = self.Q[head]
            self.K = self.K[head]
            self.V = self.V[head]
            q = x @ self.Q
            k = x @ self.K
            v = x @ self.V
            self.A = (q @ k.t())/self.out_channels**0.5
            self.A = F.softmax(self.A,dim=1)
            out = self.A @ v
            if head == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all,out),dim=1)
        
        muti_head_out = self.line(out_all)
        return muti_head_out+ x
            

class NewGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels,dropout,alpha,concat=True,head=8):
        super(NewGAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.alpha = alpha
        self.head = head
        self.concat = concat
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(in_channels, out_channels)) for _ in range(head)])
        self.a = nn.ParameterList([nn.Parameter(torch.Tensor(2 * out_channels, 1)) for _ in range(head)])
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
        
    def reset_parameters(self):
        for w in self.W:
            nn.init.xavier_uniform_(w.data, gain=1.414)
        for a in self.a:
            nn.init.xavier_uniform_(a.data, gain=1.414)

    def forward(self, x, edge_index):
        # 加上self-loop
        edge_index_with_self_loops, _ = pyg_utils.add_self_loops(edge_index, num_nodes=x.size(0))
        # 获得加上self-loop的邻接矩阵
        adj_matrix = to_dense_adj(edge_index_with_self_loops, max_num_nodes=x.size(0))[0]

        outputs = []
        for head in range(self.head):
            h = torch.mm(x, self.W[head])  # [N, out_channels]
            N = h.size(0)  # 节点个数
            input_concat = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_channels)# [N, N, 2 * out_channels]
            e = self.leakyrelu(torch.matmul(input_concat, self.a[head]).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e)  # 没连接的边置为负无穷
            attention = torch.where(adj_matrix > 0, e, zero_vec)  # [N, N]
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            out = torch.matmul(attention, h)  # [N, N] @ [N, out_channels] ==> [N, out_channels]
            outputs.append(out)

        if self.concat:
            out = torch.cat(outputs, dim=1)  # [N, out_channels * head]
        else:
            out = torch.mean(torch.stack(outputs), dim=0)
        
        return F.softmax(out,dim=1)
        
        
        


cora_dataset = Planetoid(root='cora', name='Cora')
citeseer_dataset = Planetoid(root='citeseer', name='CiteSeer')
pubmed_dataset = Planetoid(root='pubmed', name='PubMed')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('now my device is:',device)
train_loader = DataLoader(cora_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(cora_dataset, batch_size=64, shuffle=False)
model=NewGAT(cora_dataset.num_features, 16, 0.5, -0.2,concat=False,head=8).to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 注意：添加data.batch以支持批处理
        loss = criterion(out[data.train_mask], data.y[data.train_mask]).to(device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

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


best_loss = float('inf')
patience = 10  # 设置耐心阈值
patience_counter = 0  # 初始化耐心计数器
for epoch in range(200):
    model.train()
    loss = train()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss))
    #早停逻辑
    # if loss < best_loss:
    #     best_loss = loss
    #     patience_counter = 0  # 重置耐心计数器
    # else:
    #     patience_counter += 1  # 增加耐心计数器
    
    # if patience_counter >= patience:
    #     print("Early stopping triggered. Stopping training.")
    #     break  # 达到耐心阈值，停止训练
    
model.eval()
test_acc = test(test_loader)
h = model(cora_dataset[0].x.to(device), cora_dataset[0].edge_index.to(device))
visualize(h, color=cora_dataset[0].y)
print('Test Accuracy: {:.4f}'.format(test_acc))#第一次我自己改的aggressgate是0.7670，第二次leakyrelu直接0.7410，

#为何我自己写的矩阵乘法版本的gcn在cora数据集上Test Accuracy: 0.7140，论文里是0.81
#GAT仅十轮就有Test Accuracy: 0.7320