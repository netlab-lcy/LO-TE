import networkx as nx 
import json
from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def transform_path_link(path, link_inv):
    path_link = []
    for k in range(len(path) - 1):
        path_link.append(link_inv[(path[k], path[k+1])])
    return path_link

def compute_distance(G, path):
    path_cost = nx.path_weight(G, path, weight='weight')
    return path_cost

# BRITE large
# topologies = ['500_0', '1000_0', '1500_0']    

# topology zoo large
# topologies = ['Deltacom', 'GtsCe', 'TataNld', 'Interoute', 'UsCarrier', 'DialtelecomCz', 'VtlWavenet2008', 'Colt', 'Ion', 'VtlWavenet2011', 'Cogentco', 'Kdl']

# AS topology
topologies = ['ASN2k']

for topo in topologies:
    print("topo:", topo)
    topo_file = open("../../data/topology/%s_topo.txt" % (topo), "r")
    link_weight_file = open("./datas/link_weight/%s.txt" % (topo), "w")
    path_file = open("./datas/path_data/%s.json" % (topo), "w") # TO BE CHECKED begore running

    content = topo_file.readline().strip().split()
    content = list(map(int, content))
    node_num = content[0]
    link_num = content[1]
    linkSet = []
    for line in topo_file.readlines():
        content = line.strip().split()
        content = list(map(int, content))
        linkSet.append([content[0]-1, content[1]-1, content[2], content[3]]) # u, v, weight, capacity

    paths_data = {}

    link_inv = {}
    link_weights = []
    for i in range(link_num):
        link_inv[(linkSet[i][0], linkSet[i][1])] = i * 2
        link_inv[(linkSet[i][1], linkSet[i][0])] = i * 2 + 1
        link_weight = linkSet[i][2] # for BRITE/AS/GEANT topology, link length: TO BE CHECKED before running
        # link_weight = int(random.uniform(10, 100)) # for topolgoy zoo: TO BE CHECKED before running
        link_weights.append(link_weight)
    weighted_link = [(linkSet[i][0], linkSet[i][1], link_weights[i]) for i in range(link_num)]
    paths_data['link_weight'] = link_weights

    # print link weight
    link_weights_str = list(map(str, link_weights))
    print(' '.join(link_weights_str), file=link_weight_file)
    print("finish writing link weight")
    
    # calculate candidate paths
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_link)
    sp_paths = dict(nx.all_pairs_dijkstra_path(G))
    
    sp_paths_link = {}
    edge_disjoint_paths_link = {}
    k_shortest_paths_link = {}
    k = 4
    for i in range(node_num):
        sp_paths_link[i] = {}
        edge_disjoint_paths_link[i] = {}
        k_shortest_paths_link[i] = {}
        for j in range(node_num):
            if i == j:
                sp_paths_link[i][j] = []
                edge_disjoint_paths_link[i][j] = []
                k_shortest_paths_link[i][j] = []
            else:
                sp_paths_link[i][j] = transform_path_link(sp_paths[i][j], link_inv)
                edge_disjoint_paths_link[i][j] = [transform_path_link(path, link_inv) for path in sorted(nx.edge_disjoint_paths(G, i, j), key=lambda path: compute_distance(G, path))]
                k_shortest_paths_link[i][j] = [transform_path_link(path, link_inv) for path in k_shortest_paths(G, i, j, k, weight='weight')]
    
    paths_data['sp_path_link'] = sp_paths_link
    paths_data['sp_path_node'] = sp_paths
    paths_data['edge_disjoint_path_link'] = edge_disjoint_paths_link
    paths_data['k_shortest_path_link'] = k_shortest_paths_link

    data_str = json.dumps(paths_data)
    print(data_str, file=path_file)

    path_file.close()
    

    