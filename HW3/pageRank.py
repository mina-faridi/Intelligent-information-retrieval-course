import networkx as nx
import sys
import csv


# takes input from the file and creates a weighted graph
def createGraph():
	G = nx.Graph()
	f = open('spider10k.txt')
	Lines = f.readlines()
	# n = int(f.readline())
	for line in Lines:
		graph_edge_list = line.split()
		print(f)
		G.add_edge(graph_edge_list[0], graph_edge_list[1])
		nx.all_neighbors(G,G.nodes[0])
	# source, dest= f.read().splitlines()
	# return G, source, dest
	return G


if __name__ == '__main__':
	mygraph = createGraph()

	result=nx.pagerank(mygraph)
	# , alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
	# print(type(result))

	with open('pr_10k.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in result.items():
		   writer.writerow([key, value])


	# nx.write_edgelist(resultgraph, "test.edgelist.csv")

	# resultgraph.to_csv('10k.csv', index=True, header=True)

	# with open('de_10k.csv', 'w') as f:
	# 	print(resultgraph, file=f)
