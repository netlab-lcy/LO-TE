import torch as th 
import numpy as np
import torch.multiprocessing as mp

from lib.model import NetworkFinetuningModel
from lib.dataloader import TEDataloader
from lib.utils import print_gpu_usage
from config.arguments import get_arg
from lib.optmodel import finetune_path_mcfsolver_throughput, finetune_path_mcfsolver_mlu, finetune_path_mcfsolver_throughput_weight
from lib.lote_utils import data_generator, generate_input_state, calc_mlu, calc_avg_weight, calc_throughput, calc_traffic_overload, remove_congestion

import time
import gurobipy as gp

def find_finetune_flow(init_path_rate, target_path_rate, path_flow, flow_num):
    with th.no_grad():
        te_diff = th.abs(target_path_rate - init_path_rate)
        te_diff_index = th.nonzero(te_diff > 1e-3, as_tuple=True)[0]
        diff_flow_index = th.unique(path_flow[te_diff_index])
        finetune_flow = th.zeros(flow_num).to(diff_flow_index.device)
        finetune_flow[diff_flow_index] = 1
    
    return finetune_flow

'''
finetune the flows' traffic assignment regarding predicted flows
'''
def finetune_fn(args):
    print("start finetuning:", args[0])
    start = time.time()
    global data
    node_num = data['node_num']
    link_num = data['link_num']
    flow_num = data['flow_num']
    link_capacities = data['link_capacities']
    global args_list
    te_objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight = args_list[args[0]]
    # te_objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight, failed_link = args_list[args[0]] # for failure evaluation
    
    
    # solve solution finetuning problem
    with gp.Env() as env:
        if te_objective == "max_throughput" or te_objective == "min_mlu":
            finetune_ret = finetune_func(node_num, link_num, flow_num, dem_rate_list, path_link_selected, path_flow_selected, path_rate_selected, link_capacities, link_usage_background, background_traffic, env=env)
            # finetune_ret = finetune_func(node_num, link_num, flow_num, dem_rate_list, path_link_selected, path_flow_selected, path_rate_selected, link_capacities, link_usage_background, background_traffic, failed_link=failed_link, env=env) # for failure evaluation
        elif te_objective == "min_weight":
            finetune_ret = finetune_func(node_num, link_num, flow_num, dem_rate_list, path_link_selected, path_flow_selected, path_rate_selected, path_weight_selected, link_capacities, link_usage_background, background_traffic, background_traffic_weight, env=env)
        else:
            raise NotImplementedError
            
    end = time.time()
    print("finetuning time:", end-start)
    print("finishing finetuning!")
    
    return  finetune_ret



if __name__ == "__main__":
    args = get_arg()
    use_gpu = True
    device = 'cuda' if args.use_cuda and th.cuda.is_available() else 'cpu'
    print("device:", device)
    mp.set_start_method('fork', force=True)
    
    # inference setting
    max_flow_num = args.max_flow_num
    T = args.T 
    alpha = args.alpha

    # model setting
    input_dim = args.input_state_dim
    head_num = args.head_num
    hidden_dim = args.hidden_dim
    
    # # initial TE solution
    init_te_solution = args.init_te_solution 
    assert init_te_solution in ['lb', 'his']
    
    # Objective
    objective = args.objective 
    pathtype = 'edge_disjoint'

    log_dir = args.log_dir
    model_load_dir = args.model_load_dir

    # load dataset
    test_data_dir = args.test_data_dir
    dataloader = TEDataloader()
    data_test = dataloader.load_datas(test_data_dir, objective, pathtype, init_te_solution)
    print("finish loading data.")

    model = NetworkFinetuningModel(input_dim, hidden_dim, head_num)
    model.load_state_dict(th.load("./models/%s/model.th" % (model_load_dir)))
    model.to(device)
    
    # model_finetune
    if objective == 'max_throughput':
        finetune_func = finetune_path_mcfsolver_throughput
    elif objective == 'min_mlu':
        finetune_func = finetune_path_mcfsolver_mlu
    elif objective == 'min_weight':
        finetune_func = finetune_path_mcfsolver_throughput_weight
    else:
        raise NotImplementedError

    finetuned_overloads = []
    finetuned_throughputs = []
    finetuned_mlus = []
    finetuned_weights = []
    opt_throughputs = []
    opt_mlus = []
    opt_weights = []
    total_demands = []
    finetune_times = []
    avg_finetune_output = 0.
    
    model.eval()
    for topo_name in data_test:
        print("topo name:", topo_name)
        test_data = data_test[topo_name]

        test_data_generator = data_generator(test_data, label=False)
        data_ind = 0
        for data in test_data_generator:
            data_ind += 1
            print("start iteration!")
            te_solution = {
                'path_rate': data['init_path_rate']}
            
            tmp_thrpt_gain = 0
            tmp_obj_gain = 0
            
            dem_rate_list = data['dem_rate'].numpy().tolist() # global variables
            path_flow_tensor = th.tensor(data['path_flow']).to(device)

            print_gpu_usage()
            # T = len(failures) # for failure evaluation
            for t in range(T):
                
                # # for failure evaluation
                # start = time.time()
                # te_solution = {
                #     'path_rate': data['init_path_rate'].clone()}
                # path_flow = data['path_flow']
                # path_num = len(data['path_flow'])
                # tmp_flow_traffic = [0.] * data['flow_num']
                # tmp_path_avail = [1] *  len(data['path_link'])
                # link_path = data['link_path']
                # for f_link in failures[t]:
                #     # print("f_link:", f_link, "link_path:", link_path[f_link])
                #     for p in link_path[f_link]:
                #         tmp_flow_traffic[path_flow[p]] += te_solution['path_rate'][p]
                #         te_solution['path_rate'][p] = 0
                #         tmp_path_avail[p] = 0
                # flag = True
                # ind = 0
                # tmp_path_num = 0
                # tmp_rate_sum = 0
                # for l in range(path_num):
                #     tmp_path_num += tmp_path_avail[l]
                #     tmp_rate_sum += te_solution['path_rate'][l]
                #     if l == path_num - 1 or path_flow[l+1] != path_flow[l]:
                #         if tmp_path_num <= 0:
                #             flag = False
                #             break
                #         while ind <= l: 
                #             if tmp_rate_sum > 0:
                #                 te_solution['path_rate'][ind] /= tmp_rate_sum
                #             else:
                #                 if tmp_path_avail[ind] > 0:
                #                     te_solution['path_rate'][ind] += tmp_flow_traffic[path_flow[ind]] 
                #                     tmp_flow_traffic[path_flow[ind]] = 0
                #             ind += 1
                #         tmp_path_num = 0
                #         tmp_rate_sum = 0
                # if not flag: 
                #     continue 
                # failures_filtered.append(failures[t])
                # print("failure evaluation modify init TE solution time(s):", time.time()-start)
                    
                        
                # Generate fine-tuned distributions for TE solution 
                start = time.time() 
                origin_input_graph, subsets = generate_input_state(data, te_solution, input_dim, max_flow_num, device)
                 
                print("generate_data_list time(s):", time.time() - start)
                start = time.time()
                link_num = data['link_num']
                flow_num = data['flow_num']
                outputs = []
                ind = 0
                for subset in subsets:
                    with th.no_grad():
                        sub_graph = origin_input_graph.subgraph(subset)
                        
                        # # for GPU memory usage testing
                        # th.cuda.empty_cache()
                        # th.cuda.reset_max_memory_allocated(device)
                        # print("before inference:")
                        # print_gpu_usage()
                        
                        y = model(sub_graph)
                        
                        # print("after inference:")
                        # print_gpu_usage()
                        
                        r = min(max_flow_num, flow_num - ind * max_flow_num)
                        output = y[link_num:link_num+r].view(-1)
                        outputs.append(output)
                        ind += 1
                    
                finetune_output = th.cat(outputs, 0)
                print("GNN inference time(s):", time.time()-start)
                
                # precalculate network state for finetuning process
                link_capacities = th.tensor(data['link_capacities']).to(device)
                dem_rate = data['dem_rate'].to(device)
                path_flow = data['path_flow']
                path_dem_rate = data['path_dem_rate'].to(device)
                path_rate = te_solution['path_rate'].to(device) 
                path_weight = data['path_weight'].to(device)
                path_link_tensor = data['path_link_tensor'].to(device)
                path_link = data['path_link']
                print("state caldulation time1:", time.time()-start)
                
                # calculate objective for original TE solution, only for debuging
                with th.no_grad():
                    origin_mlu = calc_mlu(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                    origin_avg_weight = calc_avg_weight(path_rate, path_dem_rate, path_weight)
                    origin_thrpt = calc_throughput(path_rate, path_dem_rate)
                    origin_traffic_overload = calc_traffic_overload(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                print("origin_throughput:", origin_thrpt, "origin_traffic_overload:", origin_traffic_overload, "origin_mlu:", origin_mlu, "origin_avg_weight:", origin_avg_weight)
                
                    
                path_throughput = path_rate * path_dem_rate
                path_traffic_weight = path_rate * path_dem_rate * path_weight
                path_link_load = (path_throughput).unsqueeze(-1) * path_link_tensor
                link_load = path_link_load.sum(axis=0)
                total_thrpt = th.sum(path_throughput).item()
                total_weight = th.sum(path_traffic_weight).item()
                
                # finetuning process 
                args_list = []
                args_list2 = [] # record the index of the sample
                sample_infos = []

                # No need to enumerate k samples. In stead, selecting the top-n flows  
                val, ind = finetune_output.topk(int(data['node_num']*alpha)) 
                finetune_flow = th.zeros_like(finetune_output).to(device)
                finetune_flow[ind] = 1
                print("select finetune flow number:", th.sum(finetune_flow))
                finetune_flow_mask = finetune_flow

                path_mask = finetune_flow_mask[path_flow_tensor] 
                path_selected = th.nonzero(path_mask, as_tuple=True)[0].cpu()
                
                path_link_selected = [path_link[i] for i in path_selected]
                path_flow_selected = [path_flow[i] for i in path_selected]
                
                path_selected = path_selected.to(device)
                path_weight_selected = path_weight.index_select(0, path_selected).cpu().numpy()
                path_rate_selected = path_rate.index_select(0, path_selected).cpu().numpy()
                link_usage_background = link_load - th.sum(path_link_load.index_select(0, path_selected), axis=0)
                background_traffic = total_thrpt - th.sum(path_throughput.index_select(0, path_selected)).item()
                background_traffic_weight = total_weight - th.sum(path_traffic_weight.index_select(0, path_selected)).item()
                
                link_usage_background = link_usage_background.to_dense().cpu().numpy()
                
                sample_infos.append((path_selected, finetune_flow))
                args=(objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight)
                # args=(objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight, failures[t]) # for failure evaluation

                args_list.append(args)
                args_list2.append((0,))
                finetune_val, finetune_path_rates, finetuned_flow_indices = finetune_fn((0,))
                print("finish solution finetuning process time:", time.time()-start)
                
                best_path_rates = finetune_path_rates
                best_obj = finetune_val
                best_path_selected = path_selected
                print("finetune_val:", finetune_val)
                
                
                # update TE solution
                for i in range(len(best_path_selected)):
                    te_solution['path_rate'][best_path_selected[i]] = best_path_rates[i]
                # te_solution['path_rate'][best_path_selected] = best_path_rates                
                
                
                end = time.time()
                print("solve finetuning problem time(s):", end-start)
                # Now we only consider single round finetune time
                finetune_times.append(end-start)
                
                
                # calculate objective for original TE solution, only for debuging
                with th.no_grad():
                    path_rate = te_solution['path_rate'].to(device)
                    finetune_mlu = calc_mlu(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                    finetune_avg_weight = calc_avg_weight(path_rate, path_dem_rate, path_weight)
                    finetune_thrpt = calc_throughput(path_rate, path_dem_rate)
                    finetune_traffic_overload = calc_traffic_overload(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                
                # failure_finetune_mlus.append(finetune_mlu) # for failure case evaluation
                print("iteration step:", t, "finetune mlu:", finetune_mlu, "finetune avg_weight:", finetune_avg_weight, "finetune thrpt:", finetune_thrpt, "finetune traffic_overload:", finetune_traffic_overload)

            path_dem_rate = data['path_dem_rate'].to(device)
            path_rate = te_solution['path_rate'].to(device) 
            path_link_tensor = data['path_link_tensor'].to(device)
            link_capacities = th.tensor(data['link_capacities']).to(device)
            path_link = data['path_link']
            if objective == 'max_throughput' or objective == "min_weight":
                start = time.time()
                path_rate = remove_congestion(path_rate, path_dem_rate, path_link_tensor, link_capacities, path_link)
                print("remove congestion time(s):", time.time()-start)

            # calculate objective for original TE solution, only for debuging
            with th.no_grad():
                finetune_mlu = calc_mlu(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                finetune_avg_weight = calc_avg_weight(path_rate, path_dem_rate, path_weight)
                finetune_thrpt = calc_throughput(path_rate, path_dem_rate)
                finetune_traffic_overload = calc_traffic_overload(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                print("final mlu:", finetune_mlu, "final avg_weight:", finetune_avg_weight, "final thrpt:", finetune_thrpt, "final traffic_overload:", finetune_traffic_overload)

            if objective == 'max_throughput':
                if 'optval' in data:
                    opt_throughputs.append(data['optval'])
                finetuned_throughputs.append(finetune_thrpt)    
                finetuned_overloads.append(finetune_traffic_overload)
                total_demands.append(th.sum(data['dem_rate']).item())
            elif objective == 'min_mlu':
                if 'optval' in data:
                    opt_mlus.append(data['optval'])
                finetuned_mlus.append(finetune_mlu)
                total_demands.append(th.sum(data['dem_rate']).item())
            elif objective == 'min_weight':
                if 'optval' in data:
                    opt_throughputs.append(data['optval'][0])
                    opt_weights.append(data['optval'][1])
                finetuned_throughputs.append(finetune_thrpt)
                finetuned_weights.append(finetune_avg_weight)
                finetuned_overloads.append(finetune_traffic_overload)
                total_demands.append(th.sum(data['dem_rate']).item())
            
            # break # for failure evaluation

            
    print("opt_throughputs:", opt_throughputs)
    print("finetuned_throughputs:", finetuned_throughputs)
    print("opt_mlus:", opt_mlus)
    print("finetuned_mlus:", finetuned_mlus)
    print("finetuned_overloads:", finetuned_overloads)
    print("total_demands:", total_demands)
    print("finetune_times:", min(finetune_times), max(finetune_times), sum(finetune_times)/len(finetune_times), finetune_times)
    
    if objective == 'min_mlu':
        if len(opt_mlus) > 0:
            print("opt_mlus:", min(opt_mlus), max(opt_mlus), sum(opt_mlus)/len(opt_mlus), opt_mlus)
        print("finetuned_mlus:", min(finetuned_mlus), max(finetuned_mlus), sum(finetuned_mlus)/len(finetuned_mlus), finetuned_mlus)
    if objective == 'min_weight':
        if len(opt_weights) > 0:
            print("opt_weights:", min(opt_weights), max(opt_weights), sum(opt_weights)/len(opt_weights), opt_weights)
        print("finetuned_weights:", min(finetuned_weights), max(finetuned_weights), sum(finetuned_weights)/len(finetuned_weights), finetuned_weights)

    if objective == 'max_throughput' or objective == 'min_weight':
        demand_satisfied_ratio = np.array(finetuned_throughputs) / np.array(total_demands)
        print("demand_satisfied_ratio:", np.min(demand_satisfied_ratio), np.max(demand_satisfied_ratio), np.mean(demand_satisfied_ratio), demand_satisfied_ratio.tolist())
        if len(opt_throughputs) > 0:
            opt_demand_satified_ratio = np.array(opt_throughputs) / np.array(total_demands)
            print("opt_demand_satified_ratio:", np.min(opt_demand_satified_ratio), np.max(opt_demand_satified_ratio), np.mean(opt_demand_satified_ratio), opt_demand_satified_ratio.tolist())

    # # for failure case evaluation
    # print("failure_finetune_mlus:", failure_finetune_mlus) 
    # print("failures_filtered:", failures_filtered)
    

    
 

   