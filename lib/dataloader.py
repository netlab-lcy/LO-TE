import json
import glob
import os
import re

class TEDataloader():
    def __init__(self):
        pass 

    def load_topo(self, data_dir):
        linkSet = []
        with open(data_dir, "r") as f:
            ind = 0
            for line in f.readlines():
                if ind == 0:
                    node_num, link_num = list(map(int, line.strip().split()))
                else:
                    link = list(map(int, line.split()))
                    link[0] -= 1
                    link[1] -= 1
                    linkSet.append(link)
                    linkSet.append([link[1], link[0], link[2], link[3]]) 
                ind += 1
        return node_num, link_num*2, linkSet

    def load_TM(self, data_dir):
        TMs = []
        with open(data_dir, "r") as f:
            for line in f.readlines():
                TM = list(map(float, line.strip().split(',')))
                TMs.append(TM)
        return TMs 
    
    def load_link_weight(self, data_dir):
         with open(data_dir, "r") as f:
            line = f.readline()
            link_weight = list(map(float, line.strip().split(' ')))
            bi_dir_link_weight = [link_weight[l//2] for l in range(len(link_weight)*2)]
            return bi_dir_link_weight
    '''
        pathtype: approach to generate candidate paths: edge_disjoint|k_shortest
    '''
    def load_datas(self, data_dir, objective, pathtype, init_te_solution):
        print("start loading data!")
        dataset_paths = glob.glob(os.path.join(data_dir, "*.json"))
    
        data_dict = {}
        for data_path in dataset_paths:
            # match filename like xxx/xxx/[xxx_dd]_xxx.json' where d indicates digit
            topoName = re.findall(".*/(.+?)_\D*.json", data_path)[0] 
            print("topoName:", topoName)
            
            if topoName not in data_dict:
                node_num, link_num, linkSet = self.load_topo("./data/topology/%s_topo.txt" % (topoName))
                
                link_weight = self.load_link_weight("./data/link_weight/%s.txt" % (topoName))

                path_link, path_flow, path_weight, flow_path_ind, link_paths = self.load_pathdata("./data/path_data/%s.json" % (topoName), node_num, link_num, link_weight, pathtype)
                te_data = self.load_tedata(data_path, objective, init_te_solution)
                
                flow_num = node_num*(node_num-1)
                edge_index = self.generate_edge_index(link_num, flow_num, path_link, path_flow)
                data_dict[topoName] = {
                    'node_num': node_num,
                    'link_num': link_num, # directional link
                    'linkSet': linkSet, 
                    'edge_index': edge_index,
                    'path_link': path_link,
                    'path_flow': path_flow,
                    'path_weight': path_weight,
                    'link_path': link_paths,
                    'flow_path_ind': flow_path_ind, # the starting index of paths 
                    'te_data': te_data,
                    'link_weight': link_weight, 
                    'flow_num': flow_num,
                }
            else:
                te_data = self.load_tedata(data_path, objective, init_te_solution)
                data_dict[topoName]['te_data'] += te_data
        return data_dict
    
    def generate_edge_index(self, link_num, flow_num, path_link, path_flow):
        # input graph state: [link_num|flow_num|path_num, input_dim]
        # path_num at last for convinience of adding paths in fintuning process in the future
        total_path_num = len(path_link)
        aggregate_edge_index = []
        for pid in range(total_path_num):
            for l in path_link[pid]:
                edge = [l, link_num + flow_num + pid]
                aggregate_edge_index.append(edge)
            edge = [link_num + flow_num + pid, link_num + path_flow[pid]]
            aggregate_edge_index.append(edge)
        return aggregate_edge_index

    def load_pathdata(self, data_path, node_num, link_num, link_weight, pathtype):
        data_file = open(data_path, "r")
        line = data_file.readline()
        data = json.loads(line)

        if pathtype == "edge_disjoint":
            path_data = data['edge_disjoint_path_link']
        elif pathtype == 'k_shortest':
            path_data = data['k_shortest_path_link']
        else:
            raise NotImplementedError
        link_paths = [[] for i in range(link_num)]
        path_links = []
        path_flows = [] # corresponding flow of each path
        path_weights = []
        flow_path_ind = []
        ind = 0
        tmp_path_ind = 0
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    continue
                flow_path_links = path_data[str(i)][str(j)]
                flow_path_ind.append(tmp_path_ind)
                tmp_path_ind += len(flow_path_links)
                for p in flow_path_links:
                    path_flows.append(ind)
                    path_links.append(p)
                    path_weight = 0.
                    for l in p: 
                        link_paths[l].append(len(path_links)-1)
                        path_weight += link_weight[l]
                    path_weights.append(path_weight)
                ind += 1
        
        return path_links, path_flows, path_weights, flow_path_ind, link_paths
    
    def load_tedata(self, data_path, objective, init_te_solution):
        datas = []
        data_file = open(data_path, "r")
        for line in data_file.readlines():
            data = json.loads(line)
            target_data = {}
            target_data['dem_rate'] = data['dem_rate']
            if init_te_solution == "lb":
                target_data['init_path_rates'] = data['lb_path_rates']
                if 'pathmcf_minlbdiff_path_rates' in data:
                    target_data['pathmcf_mindiff_path_rates'] = data['pathmcf_minlbdiff_path_rates']
            elif init_te_solution == "his":
                target_data['init_path_rates'] = data['his_path_rates']
                if 'pathmcf_minhisdiff_path_rates' in data:
                    target_data['pathmcf_mindiff_path_rates'] = data['pathmcf_minhisdiff_path_rates']
            if objective == "min_mlu":
                if 'pathmcf_MLU' in data:
                    target_data['optval'] = (data['pathmcf_MLU'])
            elif objective == "max_throughput":
                if 'pathmcf_thrpt' in data:
                    target_data['optval'] = (data['pathmcf_thrpt'])
            elif objective == "min_weight":
                if 'pathmcf_thrpt' in data and 'pathmcf_weight' in data:
                    target_data['optval'] = (data['pathmcf_thrpt'], data['pathmcf_weight'])
            datas.append(target_data)
        return datas

