'''
generate initial and FIne-tune TE solution for MCF routing (maximize throughput)
'''
from lib.opt_model import path_mcfsolver, path_mcfsolver_throughput, path_mcfsolver_minpathdiff_throughput
import gurobipy as gp
import json
import numpy as np
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


# constrain the tm range to reduce the memory usage for super large topologies, TO BE CHECKED BEFORE RUNNING
TM_info = {'traffic_burst': tm_range, 'hose': tm_range, 'gravity': None} 
# TM_info = {'real': (0, 672)} # Real TM data

# target TM types in the experiment
TM_types = ['traffic_burst', 'hose',  'real'][:1] # TO BE CHECKED BEFORE RUNNING
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
    
    # TE solution for gravity model
    dem_rate_allone = np.ones(nodeNum * nodeNum - nodeNum)
    start = time.time()
    objval_allone, path_rates_allone = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate_allone, candidate_pathSet, linkSet)
    end = time.time()
    print("LB init solution time:", end - start)
    
    for tm_type in TM_types:
        print("TM type:", tm_type)

        # calculate scale factor for different TM types
        if tm_type == "traffic_burst" or tm_type == "gravity":
            dem_rate = np.array(demRates['gravity'][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            scale_factor = 1 / objval
        elif tm_type == "hose" or tm_type == "real":
            dem_rate = np.array(demRates[tm_type][0])
            objval, path_rates = path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
            scale_factor = 1 / objval
        else:
            # We have deprecated uniform TM 
            raise NotImplementedError
        
        # calculate historical init TE solution
        dem_rate = np.array(demRates[tm_type][0]) * scale_factor
        opt_mlu, path_rates_his = path_mcfsolver_throughput(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
        
        if labeled:
            result_file = open("./datas/maxthrpt/%s_label/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
        else:
            result_file = open("./datas/maxthrpt/%s_unlabel/%s_%s.json" % (topology_type, topo_name, tm_type), "w")
        
        tm_num = len(demRates[tm_type])
        for i in range(tm_num):
            print("tm_num:", i)
            
            results = {}
            dem_rate = np.array(demRates[tm_type][i])
            dem_rate = dem_rate * scale_factor

            results['dem_rate'] = dem_rate.tolist()
            results['lb_path_rates'] = path_rates_allone
            results['his_path_rates'] = path_rates_his 
            
            if labeled:
                # optimal path mcf
                start = time.time()
                objval, path_rates = path_mcfsolver_throughput(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet)
                end = time.time()
                print("path mcf objval:", objval, "time:", end - start)
                results['pathmcf_thrpt'] = objval
                results['pathmcf_path_rates'] = path_rates
                
                with gp.Env() as env:
                    results['pr_gap'] = pr_gap
                    print("pr_gap:", pr_gap)
                    # Fine-tuning solution for LB init solution
                    start = time.time()
                    # LP_relax+rounding solution, for optimal, use the same parameter with path_mcfsolver_minpathdiff_throughput_optimal
                    objval, scale_factor_val, path_rates, flow_diffs = path_mcfsolver_minpathdiff_throughput(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, path_rates_allone, results['pathmcf_thrpt'] * (1-pr_gap), scale_factor_lb=1, scale_factor_ub=1, env=env) 
                    print("scale_factor_val:", scale_factor_val)
                    end = time.time()
                    print("min path diff:", objval, "time:", end-start)
                    diff_path = 0
                    for k in range(demNum):
                        if flow_diffs[k] > 1e-4:
                            diff_path += 1
                    print("min path diff lb path ratio:", diff_path / demNum)
                    pathmcf_gravitydiff_ratios.append(diff_path / demNum)
                    results['pathmcf_minlbdiff_path_rates'] = path_rates
                    results['pathmcf_minlbdiff_pathdiff_ratio'] = diff_path / demNum
                    
                    # Fine-tuning solution for Historical init solution
                    start = time.time()
                    # LP_relax+rounding solution, for optimal, use the same parameter with path_mcfsolver_minpathdiff_throughput_optimal
                    objval, scale_factor_val, path_rates, flow_diffs = path_mcfsolver_minpathdiff_throughput(nodeNum, linkNum, demNum, demands, dem_rate, candidate_pathSet, linkSet, path_rates_his, results['pathmcf_thrpt'] * (1-pr_gap), scale_factor_lb=1, scale_factor_ub=1, env=env) 
                    print("scale_factor_val:", scale_factor_val)
                    end = time.time()
                    print("min path diff:", objval, "time:", end-start)
                    diff_path = 0
                    for k in range(demNum):
                        if flow_diffs[k] > 1e-4:
                            diff_path += 1
                    print("min path diff his path ratio:", diff_path / demNum)
                    pathmcf_gravitydiff_ratios.append(diff_path / demNum)
                    results['pathmcf_minhisdiff_path_rates'] = path_rates
                    results['pathmcf_minhisdiff_pathdiff_ratio'] = diff_path / demNum
                    

            print(json.dumps(results), file=result_file)
        result_file.close()



