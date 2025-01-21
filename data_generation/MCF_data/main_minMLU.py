'''
generate initial TE solution for MCF routing (maximize throughput)
'''
from lib.opt_model import path_mcfsolver, path_mcfsolver_minpathdiff, path_mcfsolver_minpathdiff_optimal
from lib.topo import ReadTopo
import gurobipy as gp
import json
import numpy as np
import sys 
import time



# BRITE large
# topologies = ['500_0', '1000_0', '1500_0']

# topologies = ['GEANT']

# topology zoo large
# topologies = ['Deltacom', 'GtsCe', 'TataNld', 'Interoute', 'UsCarrier', 'DialtelecomCz', 'VtlWavenet2008', 'Colt', 'Ion', 'VtlWavenet2011', 'Cogentco', 'Kdl']

# AS topology
topologies = ['ASN2k']

topology_type = "AS" # TO BE CHECKED BEFORE RUNNING
labeled = True # False for USL data, True for SL data, TO BE CHECKED BEFORE RUNNING
if labeled:
    tm_range = (0, 120)
else:
    tm_range = (200, 300) 


# constrain the tm range to reduce the memory usage for super large topologies，TO BE CHECKED BEFORE RUNNING
TM_info = {'traffic_burst': tm_range, 'hose': tm_range, 'gravity': None}  # Normal data
# TM_info = {'real': (672, 1344)} # Real TM data

# target TM types in the experiment
TM_types = ['traffic_burst', 'hose', 'gravity', 'uniform', 'real'][1:2]  # TO BE CHECKED BEFORE RUNNING
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

    # load candidiate path set for path_mcf
    candidate_paths_link = pathinfo['edge_disjoint_path_link']
    # candidate_paths_link = pathinfo['k_shortest_path_link']

    candidate_pathSet = []
    min_path_num = 1e6
    for d in demands:
        min_path_num = min(min_path_num, len(candidate_paths_link[str(d[0])][str(d[1])]))
        candidate_pathSet.append(candidate_paths_link[str(d[0])][str(d[1])])
    print("min_path_num:", min_path_num)
    
    # single path routing
    sp_path_rates = []
    for k in range(demNum):
        sp_path_rate = [0] * len(candidate_pathSet[k])
        sp_path_rate[0] = 1
        sp_path_rates.append(sp_path_rate)
    
    
    # TE solution for gravity model
    dem_rate_allone = np.ones(nodeNum * nodeNum- nodeNum)
    start = time.time()
    objval_allone, path_rates_allone = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate_allone, candidate_pathSet, linkSet)
    end = time.time()
    print("LB init solution time:", end - start)

    
    for tm_type in TM_types:
        print("TM type:", tm_type)

        # calculate scale factor for different TM types
        # for MLU scale factor is unnecessary, but we use it to avoid numerical issue in gurobi
        if tm_type == "traffic_burst" or tm_type == "gravity":
            dem_rate = np.array(demRates['gravity'][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            scale_factor = 1 / objval
        elif tm_type == "hose" or tm_type == "real":
            dem_rate = np.array(demRates[tm_type][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            scale_factor = 1 / objval
            print("scale_factor:", scale_factor)
        else:
            # We have deprecated uniform TM 
            raise NotImplementedError
        
        start = time.time()
        # calculate historical init TE solution
        dem_rate = np.array(demRates[tm_type][0]) * scale_factor
        opt_mlu, path_rates_his = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
        print("historical init solution time:", time.time() - start, "objval:", opt_mlu)

        if labeled:
            result_file = open("./datas/minMLU/%s_label/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
        else:
            result_file = open("./datas/minMLU/%s_unlabel/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
            
        tm_num = len(demRates[tm_type])
        for i in range(tm_num):
            results = {}
            dem_rate = np.array(demRates[tm_type][i]) 
            dem_rate = dem_rate * scale_factor
            
            results['dem_rate'] = dem_rate.tolist()
            results['lb_path_rates'] = path_rates_allone
            results['his_path_rates'] = path_rates_his 
            
            
            if labeled:
                # optimal path mcf
                start = time.time()
                objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, dp_ratio=1.) # for testing
                end = time.time()
                if objval < 0:
                    continue
                print("path mcf objval:", objval, "time:", end - start)
                
                results['pathmcf_MLU'] = objval
                results['pathmcf_path_rates'] = path_rates
            
            
                print("pr_gap:", pr_gap)
                with gp.Env() as env:
                    # for demand pinning
                    # start = time.time()
                    # objval_dp, path_rates_dp = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, dp_ratio=0.1, env=env)
                    # end = time.time()
                    # print("path mcf dp objval:", objval, "time:", end - start, "pr:", objval_dp / results['path_mcf_MLU'])
                    # prs.append(objval_dp / results['path_mcf_MLU'])
                    
                    
                    # Fine-tuning solution for shortest path routing
                    # start = time.time()
                    # objval, path_rates, flow_diffs = path_mcfsolver_minpathdiff(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, sp_path_rates, results['pathmcf_MLU'] * (1+pr_gap), env=env)
                    # end = time.time()
                    # print("mlu:", objval, "pr:", objval / results['pathmcf_MLU'], "satisfied:", objval <= results['pathmcf_MLU'] * (1+pr_gap))
                    # diff_path = 0
                    # for k in range(demNum):
                    #     if flow_diffs[k] > 1e-4:
                    #         diff_path += 1
                    # print("min path diff sp path ratio:", diff_path / demNum)
                    # pathmcf_spdiff_ratios.append(diff_path / demNum)
                    # results['pathmcf_minspdiff_path_rates'] = path_rates
                    # results['pathmcf_minspdiff_pathdiff_ratio'] = diff_path / demNum
                    
                    
                    # Fine-tuning solution for LB init solution
                    start = time.time()
                    objval, path_rates, flow_diffs = path_mcfsolver_minpathdiff(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, path_rates_allone, results['pathmcf_MLU'] * (1+pr_gap), env=env)
                    end = time.time()
                    print("mlu:", objval, "pr:", objval / results['pathmcf_MLU'], "satisfied:", objval <= results['pathmcf_MLU'] * (1+pr_gap), "time:", end - start)
                    diff_path = 0
                    for k in range(demNum):
                        if flow_diffs[k] > 1e-4:
                            diff_path += 1
                    print("min path diff allone_mcf path ratio:", diff_path / demNum)
                    pathmcf_gravitydiff_ratios.append(diff_path / demNum)
                    results['pathmcf_minlbdiff_path_rates'] = path_rates
                    results['pathmcf_minlbdiff_pathdiff_ratio'] = diff_path / demNum
                    
                    
                    # Fine-tuning solution for Historical init solution
                    start = time.time()
                    objval, path_rates, flow_diffs = path_mcfsolver_minpathdiff(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, path_rates_his, results['pathmcf_MLU'] * (1+pr_gap), env=env)
                    end = time.time()
                    print("mlu:", objval, "pr:", objval / results['pathmcf_MLU'], "satisfied:", objval <= results['pathmcf_MLU'] * (1+pr_gap), "time:", end - start)
                    diff_path = 0
                    for k in range(demNum):
                        if flow_diffs[k] > 1e-4:
                            diff_path += 1
                    print("min path diff historical_mcf path ratio:", diff_path / demNum)
                    pathmcf_gravitydiff_ratios.append(diff_path / demNum)
                    results['pathmcf_minhisdiff_path_rates'] = path_rates
                    results['pathmcf_minhisdiff_pathdiff_ratio'] = diff_path / demNum
                    
                    
            print(json.dumps(results), file=result_file)
        result_file.close()
# print("prs:", prs)
# print("path mcf sp path diff ratios:", pathmcf_spdiff_ratios)
# print("path mcf pathmcf path diff ratios:", pathmcf_gravitydiff_ratios)
# print("path mcf dp path diff ratios:", pathmcf_dpdiff_ratios)