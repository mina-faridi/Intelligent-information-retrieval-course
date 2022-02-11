import networkx as nx
import sys


def dead_ends(G):
    dead_ends = {}
    ends = []
    # Find all ends of graphs
    for node in G.nodes:
        if len(list(nx.neighbors(G,node))) == 0:
            dead_ends[node] = True
            ends.append(node)
        else:
            dead_ends[node] = False

    # print(dead_ends)
    # Copy edges:
    local_edges = {}
    for node in G.nodes:
        local_edges[node] = list(nx.neighbors(G,node))

    # Starts from ends to crawl back the graph
    for node in ends:
        # For each end of graph found
        continue_to_crawl = True  # Sets a boolean to stop the while loop
        current_node = node  # Sets current node to node
        while continue_to_crawl:
            if dead_ends[current_node] == True and len(local_edges[current_node]) == 1:
                next_node = local_edges[current_node][0]  # finds next node
                dead_ends[next_node] = True  # Sets dead_ends for the next node as a dead end
                del local_edges[current_node]  # removes key for current node
                local_edges[next_node].remove(current_node)  # Removes current node from the next node edge list
                current_node = next_node

            elif len(local_edges[current_node]) > 1:
                break
    	with open('pr_10k.csv', 'w') as csv_file:

		writer = csv.writer(csv_file)

		for key, value in result.items():
		   writer.writerow([key, value])


if __name__ == '__main__':
    mygraph = nx.read_edgelist("spider800k.txt", comments='#', delimiter=None, create_using=None, nodetype=None,
                               data=True, edgetype=None, encoding='utf-8')
    dead_ends(mygraph)


