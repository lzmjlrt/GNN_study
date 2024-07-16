import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric 
from torch_geometric.datasets import WebKB

# 指定数据集的根目录
root = './data/WebKB'
# 选择WebKB数据集中的一个部分，例如Cornell
name = 'cornell'

# 加载数据集
dataset = WebKB(root=root, name=name)

# 获取数据
data = dataset[0]
#print(data)
G = torch_geometric.utils.to_networkx(data, to_undirected=False)

# 绘制图
nx.draw(G, with_labels=True)
plt.show()
#建立图的随机邻接矩阵
M = torch_geometric.utils.to_dense_adj(data.edge_index)
M_numpy = M.numpy()  # 转换为numpy数组
is_symmetric = np.array_equal(M_numpy, M_numpy.T)
print('M_numpy is symmetric:', is_symmetric)
#rank vector的power iteration method 
rank = torch.ones(M.size(1),1)
print(rank.shape)
# 把rank里每个值改为1/n，n为节点个数
rank = rank/data.num_nodes
M=M.squeeze(0)
#换为The Google Matrix G
beta = 0.8
M=M/(M.sum(dim=0,keepdim=True)+1e-10)
# 计算并打印每一列的和
M = beta*M + (1-beta)*(torch.ones(M.size())/data.num_nodes)
print('M:',M)
for i in range(250):
    r_1 = torch.mm(M,rank)
    #r_1 = r_1/torch.sum(r_1)
    if torch.sum(torch.abs(r_1-rank))<1e-5:
        print('converged')
        print('in iteration {}.the rank vector is:{}'.format(i,r_1))
        rank = r_1
        break
    else:
        rank = r_1
# # 计算特征值和特征向量
# eigenvalues, eigenvectors = torch.linalg.eig(M)

# # 将特征值从复数转换为实数部分
# eigenvalues = eigenvalues.real

# # 找到最接近1的特征值的索引
# index = torch.argmin(torch.abs(eigenvalues - 1))

# # 获取对应的特征向量，并进行归一化
# target_eigenvector = eigenvectors[:, index].real
# target_eigenvector = target_eigenvector / torch.sum(target_eigenvector)

# print(f"特征值为1的特征向量（归一化后）: {target_eigenvector}")
# rank = target_eigenvector
#按照rank值对每个node进行排序，按照rank值从大到小排序
print('Rank:', rank)
sorted_nodes = torch.argsort(rank.squeeze(), descending=True)
print('Nodes sorted by rank:', sorted_nodes)
top_10_indices = sorted_nodes[:10]

# 打印前十个节点及其标签和所有属性
for idx in top_10_indices:
    attributes = data.x[idx]  # 假设节点的所有属性存储在这里
    attributes_str = ', '.join(str(attribute.item()) for attribute in attributes)  # 将所有属性转换为字符串
    print(f'Node: {idx}, Label: {data.y[idx].item()}')