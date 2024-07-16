#复现Relational Classification / Probabilistic Relational Classifier
import networkx as nx
import numpy as np
import torch
import torch_geometric

G = nx.karate_club_graph()

#把里面的一些节点的当作有标签节点，剩下的当作无标签节点，对于有标签的节点，给他固定概率为1或0
#对应无标签节点，给他固定概率为0.5
probablitity = {}
for node in G.nodes():
    probablitity[node] = 0.5
    
num_labled_nodes = int(0.5*G.number_of_nodes())
labled_nodes = np.random.choice(G.nodes(), size=num_labled_nodes, replace=False)

for node in labled_nodes:
    club = G.nodes[node]['club']
    if club == 'Mr. Hi':
        probablitity[node] = 1
    elif club == 'Officer':
        probablitity[node] = 0

#对没有标签的节点，给他一个固定的概率为0.5
for node in G.nodes():
    if node not in labled_nodes:
        probablitity[node] = 0.5
new_probablitity = []
old_probablitity = probablitity.copy()    
for i in range(1000):
    new_probablitity = probablitity.copy()
    # 按照node的顺序，对每个节点进行更新
    for node in sorted(G.nodes()):
        if node in labled_nodes:
            continue
        else:
            neighbors = list(G.neighbors(node))
            new_probablitity[node] = sum([probablitity[neighbor] for neighbor in neighbors])/len(neighbors)
    # 检查概率是否收敛
    if new_probablitity == old_probablitity:
        print('in iter {} converd'.format(i))
        break
    old_probablitity = new_probablitity.copy()
    probablitity = new_probablitity
                
                
#打印所有节点的概率
for node in G.nodes():
    print(node, probablitity[node])
                
#如果节点的概率大于0.5，那么就认为是Mr. Hi，否则就认为是Officer
for node in G.nodes():
    if probablitity[node] > 0.5:
        G.nodes[node]['predict_club'] = 'Mr. Hi'
    else:
        G.nodes[node]['predict_club'] = 'Officer'
        
#按照predict_club和club的比较，计算准确率
correct = 0
for node in G.nodes():
    if G.nodes[node]['predict_club'] == G.nodes[node]['club']:
        correct += 1
        
accuracy = correct/G.number_of_nodes()
print(accuracy)
#展示predict_club分类的图
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
# 创建一个图形和两个子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
origin_colors = []
for node in G.nodes():
    if G.nodes[node]['club'] == 'Mr. Hi':
        origin_colors.append('red')
    else:
        origin_colors.append('blue')
# 在第一个子图上绘制原始数据集
nx.draw(G, pos, node_color=origin_colors, with_labels=True, ax=axs[0])
axs[0].set_title('Original Dataset')
colors = []
for node in G.nodes():
    if G.nodes[node]['predict_club'] == 'Mr. Hi':
        colors.append('red')
    else:
        colors.append('blue')
# 在第二个子图上绘制分类后的数据集
nx.draw(G, pos, node_color=colors, with_labels=True, ax=axs[1])
axs[1].set_title('Classified Dataset')

plt.show()