import networkx as nx
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt 
G = nx.Graph()

G.add_node(1)

G.add_nodes_from([
	(4, {"color": "red"}),
	(5, {"color": "green"}),
])

G.add_edge(1, 2)

nx.draw(G)
plt.show()