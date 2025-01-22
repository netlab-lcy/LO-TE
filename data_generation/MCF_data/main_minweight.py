from lib.opt_model import path_mcfsolver, path_mcfsolver_minweight, path_mcfsolver_minpathdiff_minweight
from lib.topo import ReadTopo
import gurobipy as gp
import json
import numpy as np
import sys
import time




# BRITE large
# topologies = ['500_0', '1000_0', '1500_0'][:2]

topologies = ['GEANT']


# topology zoo large
# topologies = ['Deltacom', 'GtsCe', 'TataNld', 'Interoute', 'UsCarrier', 'DialtelecomCz', 'VtlWavenet2008', 'Colt', 'Ion', 'VtlWavenet2011', 'Cogentco', 'Kdl']

topology_type = "topology_zoo_middle" # TO BE CHECKED BEFORE RUNNING
labeled = False # False for USL data, True for SL data, TO BE CHECKED BEFORE RUNNING
if labeled:
    tm_range = (0, 120)
else:
    tm_range = (1000, 2000)


# constrain the tm range to reduce the memory usage for super large topologies # TO BE CHECKED BEFORE RUNNING
# TM_info = {'traffic_burst': tm_range, 'hose': tm_range, 'gravity': None} 
TM_info = {'real': (0, 672)} # Real TM data

# target TM types in the experiment
TM_types = ['traffic_burst', 'hose', 'real'][-1:] # TO BE CHECKED BEFORE RUNNING
data_dir = "../../data" # topo and TM data dir

pr_gap = 0.01

prs = []
pathmcf_spdiff_ratios = []
pathmcf_gravitydiff_ratios = []
pathmcf_dpdiff_ratios = []

for topo_name in topologies:
    print("topo_name:", topo_name)
    topoloader = ReadTopo(data_dir, topo_name, TM_info)
    nodeNum, linkNum, linkSet, demNum, demands, demRates, pathinfo, cMatrix, wMatrix, MAXWEIGHT = topoloader.read_info()
    link_capacities = [linkSet[i//2][3] for i in range(linkNum * 2)]
    link_capacities = np.array(link_capacities)

    link_weight = pathinfo['link_weight']
    for i in range(linkNum):
        wMatrix[linkSet[i][0]][linkSet[i][1]] = link_weight[i]
        wMatrix[linkSet[i][1]][linkSet[i][0]] = link_weight[i]
        linkSet[i][2] = link_weight[i]
    
    # load candidiate path set for path_mcf
    candidate_paths_link = pathinfo['edge_disjoint_path_link']
    # candidate_paths_link = pathinfo['k_shortest_path_link']

    candidate_pathSet = []
    total_path_num = 0
    for d in demands:
        candidate_pathSet.append(candidate_paths_link[str(d[0])][str(d[1])])
        total_path_num += len(candidate_paths_link[str(d[0])][str(d[1])])

    # shortest path routing
    sp_path_rates = []
    for k in range(demNum):
        sp_path_rate = [0] * len(candidate_pathSet[k])
        sp_path_rate[0] = 1
        sp_path_rates.append(sp_path_rate)
    
    for tm_type in TM_types:
        print("TM type:", tm_type)
        # calculate scale factor and init TE solution for different TM types
        # for MLU scale factor is unnecessary, but we use it to avoid numerical issue in gurobi
        if tm_type == "traffic_burst" or tm_type == "gravity":
            dem_rate = np.array(demRates['gravity'][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            normal_factor = 1 / objval
        elif tm_type == "hose" or tm_type == "real":
            # scale_factor using hose TM 0 
            dem_rate = np.array(demRates[tm_type][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            normal_factor = 1 / objval
        else:
            # We have deprecated uniform TM 
            raise NotImplementedError
        # calculate historical init TE solution
        dem_rate = dem_rate * normal_factor
        opt_thrpt, opt_weight, path_rates_his = path_mcfsolver_minweight(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, wMatrix, MAXWEIGHT,  env=None) 
       
        if labeled:
            result_file = open("./datas/minweight/%s_label/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
        else:
            result_file = open("./datas/minweight/%s_unlabel/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
        tm_num = len(demRates[tm_type])
        for i in range(tm_num):
            results = {}
            dem_rate = np.array(demRates[tm_type][i]) # for testing
            
            dem_rate = dem_rate * normal_factor 
            results['dem_rate'] = dem_rate.tolist()
            results['his_path_rates'] = path_rates_his

            if labeled:
                # optimal path mcf
                start = time.time()
                opt_thrpt, opt_weight, path_rates = path_mcfsolver_minweight(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, wMatrix, MAXWEIGHT,  env=None) 
                end = time.time()
                print("path_wmcf opt_thrpt:", opt_thrpt, "opt_weight:", opt_weight, "time:", end-start)
                results['pathmcf_weight'] = opt_weight
                results['pathmcf_thrpt'] = opt_thrpt
                results['pathmcf_path_rate'] = path_rates
                
                print("pr_gap:", pr_gap)
                with gp.Env() as env:
                    # Fine-tuning solution for solutions of init gravity TM
                    start = time.time()
                    objval, path_rates, flow_diffs = path_mcfsolver_minpathdiff_minweight(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, path_rates_his, results['pathmcf_thrpt']*(1-pr_gap), results['pathmcf_weight']*(1+pr_gap), wMatrix, MAXWEIGHT, env=env)
                    end = time.time()
                    print("min path diff:", objval)
                    diff_path = 0
                    for k in range(demNum):
                        if flow_diffs[k] > 1e-4:
                            diff_path += 1
                        elif flow_diffs[k] > 0:
                            print("exception diff:", flow_diffs[k])
                    print("min path diff initgravity path ratio:", diff_path / demNum, "time:", end-start)
                    pathmcf_gravitydiff_ratios.append(diff_path / demNum)
                    results['pathmcf_minhisdiff_path_rates'] = path_rates
                    results['pathmcf_minhisdiff_pathdiff_ratio'] = diff_path / demNum

            print(json.dumps(results), file=result_file)
        result_file.close()























