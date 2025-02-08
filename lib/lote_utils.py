import torch as th
import time
from torch_geometric.data import Data, DataLoader
import torch_sparse


'''
flattening path rates of the flows
'''
def flatten(vec):
    ret_vec = []
    for i in vec:
        ret_vec += i
    return ret_vec

def generate_path_link_tensor(path_link, link_num):
    path_num = len(path_link)
    indices = [[],[]]
    vals = []
    for i in range(path_num):
        for j in path_link[i]:
            indices[0].append(i)
            indices[1].append(j)
            vals.append(1)
    return th.sparse_coo_tensor(indices, vals, (path_num, link_num))


'''
Initial data generation for each topology
'''
def data_generator(data, label=False):
    node_num = data['node_num']
    link_num = data['link_num']
    linkSet = data['linkSet']
    flow_num = data['flow_num']
    link_weight = data['link_weight']
    te_datas = data['te_data']
    path_link = data['path_link']
    path_flow = data['path_flow']
    path_weight = data['path_weight']
    link_path = data['link_path']
    flow_path_ind = data['flow_path_ind']
    start = time.time()
    path_link_tensor = generate_path_link_tensor(path_link, link_num)
    end = time.time()
    print("generate path tensor time(s):", end-start)
    
    
    
    opt_data = {}
    opt_data['node_num'] = node_num
    opt_data['link_num'] = link_num 
    opt_data['flow_num'] = flow_num 
    opt_data['edge_index'] = th.tensor(data['edge_index']).t().contiguous()
    opt_data['link_capacities'] = [linkSet[i][3] for i in range(link_num)]
    opt_data['link_weight'] = th.tensor(link_weight)
    path_num = len(path_link)
    
    print("te_data number:", len(te_datas))
    for te_data in te_datas:
        # for no solution case (caused by numerical issue of gurobi), to be removed
        if te_data['dem_rate'][0] < 0:
            continue
        
        opt_data['link_path'] = link_path 
        opt_data['path_link'] = path_link
        opt_data['path_flow'] = path_flow
        opt_data['path_weight'] = th.tensor(path_weight)
        opt_data['flow_path_ind'] = flow_path_ind
        opt_data['path_link_tensor'] = path_link_tensor.detach()
        print("finish copy data!")
        
        opt_data['dem_rate'] = th.tensor(te_data['dem_rate'])
        opt_data['path_dem_rate'] = opt_data['dem_rate'][path_flow]
        
        init_path_rates = te_data['init_path_rates']
        opt_data['init_path_rate'] = th.tensor(flatten(init_path_rates))
        
        if 'optval' in te_data:
            opt_data['optval'] = te_data['optval']
        if label:
            target_path_rates = te_data['pathmcf_mindiff_path_rates']
            opt_data['target_path_rate'] = th.tensor(flatten(target_path_rates))   
        
        opt_data['flow_info'] = []
        ind = 0
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    continue
                opt_data['flow_info'].append({'src': i, 'dst': j})
                ind += 1
        
        yield opt_data
        
def generate_input_state(data, te_solution, input_dim, max_flow_num, device):
    node_num = data['node_num']
    link_num = data['link_num']
    flow_num = data['flow_num']
    flow_info = data['flow_info']
    flow_path_ind = data['flow_path_ind']
    edge_index = data['edge_index']
    link_capacities = data['link_capacities']
    link_weight = data['link_weight']
    dem_rate = data['dem_rate'].to(device)
    path_weight = data['path_weight'].to(device)
    path_flow = data['path_flow']
    path_dem_rate = data['path_dem_rate'].to(device)
    path_rate = te_solution['path_rate'].to(device)
    path_link_tensor = data['path_link_tensor'].to(device)
    path_link = data['path_link']
    path_num = len(path_link)
    link_capacities = th.tensor(link_capacities).to(device)
    
    # calculate link utilities
    link_util = ((path_rate * path_dem_rate).unsqueeze(-1) * path_link_tensor).sum(axis=0).to_dense() / link_capacities
    
    x = th.zeros((link_num + flow_num + path_num, input_dim))
    x[:link_num, 0] = 1
    x[link_num:link_num+flow_num, 1] = 1
    x[link_num+flow_num:, 2] = 1
    # embed topo graph state
    x[:link_num, 3] = link_weight / th.max(link_weight)
    x[:link_num, 4] = link_util.cpu().detach()
    x[link_num:link_num+flow_num, 5] = (dem_rate / th.max(dem_rate))
    x[link_num+flow_num:, 6] = path_rate.cpu().detach()
    x[link_num+flow_num:, 7] = path_weight.cpu().detach()

    origin_graph = Data(x=x, edge_index=edge_index).to(device)
    link_set = th.arange(link_num)
    subsets = []
    for i in range(0, flow_num, max_flow_num):
        if i+max_flow_num>=flow_num:
            path_set = th.arange(link_num+flow_num+flow_path_ind[i], link_num+flow_num+path_num)
            flow_set = th.arange(link_num+i, link_num+flow_num)
        else:
            path_set = th.arange(link_num+flow_num+flow_path_ind[i], link_num+flow_num+flow_path_ind[i+max_flow_num])
            flow_set = th.arange(link_num+i, link_num+i+max_flow_num)
        subset = th.cat([link_set, flow_set, path_set], axis=0).to(device)
        subsets.append(subset)

    return origin_graph, subsets



# Calculate network performance 
def calc_mlu(path_rate, path_dem_rate, path_link_tensor, link_capacities):
    link_util = ((path_rate * path_dem_rate).unsqueeze(-1) * path_link_tensor).sum(axis=0).to_dense() / link_capacities
    mlu = th.max(link_util).cpu().detach().item()
    return mlu

def calc_avg_weight(path_rate, path_dem_rate, path_weight):
    total_weight = th.sum(path_rate * path_dem_rate * path_weight).cpu().detach().item()
    total_thrpt = th.sum(path_dem_rate * path_rate).cpu().detach().item()
    return total_weight / total_thrpt

def calc_throughput(path_rate, path_dem_rate):
    total_thrpt = th.sum(path_dem_rate * path_rate).cpu().detach().item()
    return total_thrpt


def calc_traffic_overload(path_rate, path_dem_rate, path_link_tensor, link_capacities):
    link_usage = ((path_rate * path_dem_rate).unsqueeze(-1) * path_link_tensor).sum(axis=0)
    traffic_overload = th.sum(th.clamp(-link_capacities + link_usage, min=0)).cpu().detach().item() # add(dense, sparse)
    return traffic_overload
    
def remove_congestion(path_rate, path_dem_rate, path_link_tensor, link_capacities, path_link):
    with th.no_grad():
        link_util = ((path_rate * path_dem_rate).unsqueeze(-1) * path_link_tensor).sum(axis=0).to_dense() / link_capacities
        path_link_util =  path_link_tensor * link_util.unsqueeze(0)
        coo_indices = path_link_util._indices()
        coo_values = path_link_util._values()
        row = coo_indices[0]
        col = coo_indices[1]
        path_link_util = torch_sparse.SparseTensor(row=row, col=col, value=coo_values, sparse_sizes=path_link_util.size())
        path_mlu = torch_sparse.max(path_link_util, dim=1)
        path_rate = path_rate / th.maximum(path_mlu, th.tensor(1.0).to(path_mlu.device))
    return path_rate