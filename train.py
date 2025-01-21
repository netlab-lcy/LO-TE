import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.multiprocessing as mp

from lib.model import NetworkFinetuningModel
from lib.dataloader import TEDataloader
from lib.utils import cleanup_dir,  print_gpu_usage
from config.arguments import get_arg
from lib.optmodel import finetune_path_mcfsolver_throughput, finetune_path_mcfsolver_mlu, finetune_path_mcfsolver_throughput_weight
from lib.lote_utils import data_generator, generate_input_state, calc_mlu, calc_avg_weight, calc_throughput, calc_traffic_overload

import time 
import gurobipy as gp



def find_finetune_flow(init_path_rate, target_path_rate, path_flow, flow_num):
    with th.no_grad():
        te_diff = th.abs(target_path_rate - init_path_rate)
        te_diff_index = th.nonzero(te_diff > 1e-3, as_tuple=True)[0]
        diff_flow_index = th.unique(path_flow[te_diff_index])
        diff_flow_num = diff_flow_index.size(0)
        finetune_flow = th.zeros(flow_num).to(diff_flow_index.device)
        finetune_flow[diff_flow_index] = 1
        finetune_label_weight = th.ones(flow_num) / max(flow_num-diff_flow_num, 1)
        finetune_label_weight[diff_flow_index] = 1 / max(diff_flow_num, 1)
    
    return finetune_flow, finetune_label_weight


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
    
    # solve solution finetuning problem
    with gp.Env() as env:
        if te_objective == "max_throughput" or te_objective == "min_mlu":
            finetune_ret = finetune_func(node_num, link_num, flow_num, dem_rate_list, path_link_selected, path_flow_selected, path_rate_selected, link_capacities, link_usage_background, background_traffic, min_path_diff=True, env=env)
        elif te_objective == "min_weight":
            finetune_ret = finetune_func(node_num, link_num, flow_num, dem_rate_list, path_link_selected, path_flow_selected, path_rate_selected, path_weight_selected, link_capacities, link_usage_background, background_traffic, background_traffic_weight, min_path_diff=True, env=env)
        else:
            raise NotImplementedError
            
    end = time.time()
    print("finetuning time:", end-start)
    print("finishing finetuning!")
    
    return  finetune_ret


if __name__ == "__main__":
    args = get_arg()
    device = 'cuda' if args.use_cuda and th.cuda.is_available() else 'cpu'
    print("device:", device)
    th.autograd.set_detect_anomaly(True)
    
    # training setting
    epoch_num = args.training_epochs
    alpha = args.alpha 
    T = args.T 
    K = args.K 
    num_sample_process = args.num_sample_process
    max_flow_num = args.max_flow_num

    # model setting
    input_dim = args.input_state_dim
    head_num = args.head_num
    hidden_dim = args.hidden_dim
    
    # # initial TE solution
    init_te_solution = args.init_te_solution 
    assert init_te_solution in ['lb', 'his']

    # Objective
    objective = args.objective 
    assert objective in ['max_throughput', 'min_mlu', 'min_weight']

    pathtype = 'edge_disjoint'
    training_mode = args.training_mode 
    assert training_mode in ['SL', 'USL']

    log_dir = args.log_dir
    model_load_dir = args.model_load_dir
    
    # load dataset
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    dataloader = TEDataloader()
    data_train = dataloader.load_datas(train_data_dir, objective, pathtype, init_te_solution)
    print("finish loading data.")

    model = NetworkFinetuningModel(input_dim, hidden_dim, head_num)
    if model_load_dir != None:
        model.load_state_dict(th.load("./models/%s/model.th" % (model_load_dir))) # load pretrain model parameters

    model.to(device)

    # model_train
    optimizer = optim.Adam(model.parameters())
    loss1_fn = nn.BCELoss(reduction="none")
    
    # model_finetune
    if objective == 'max_throughput':
        finetune_func = finetune_path_mcfsolver_throughput
    elif objective == 'min_mlu':
        finetune_func = finetune_path_mcfsolver_mlu
    elif objective == 'min_weight':
        finetune_func = finetune_path_mcfsolver_throughput_weight
    else:
        raise NotImplementedError

    
    loss_train = []

    for epoch in range(epoch_num):
        model.train()
        for topo_name in data_train:
            print("epoch:", epoch, "topo name:", topo_name)
            train_data = data_train[topo_name]
            if training_mode == 'SL':
                train_data_generator = data_generator(train_data, label=True)
            else:
                train_data_generator = data_generator(train_data, label=False)
            data_ind = 0
            for data in train_data_generator:
                data_ind += 1
                print("data_ind:", data_ind)
                print("start iteration!")
                te_solution = {
                    'path_rate': data['init_path_rate']}
                
                dem_rate_list = data['dem_rate'].numpy().tolist() 
                path_flow_tensor = th.tensor(data['path_flow']).to(device)

                for t in range(T):
                    # Generate refined flow distributions for initial TE solution 
                    start = time.time()
                    origin_input_graph, subsets = generate_input_state(data, te_solution, input_dim, max_flow_num, device) 
                    print("generate_data_list time(s):", time.time() - start)

                    start = time.time()
                    link_num = data['link_num']
                    flow_num = data['flow_num']
                    outputs = []
                    ind = 0
                    for subset in subsets:
                        print("ind:", ind)
                        print_gpu_usage()
                        sub_graph = origin_input_graph.subgraph(subset)
                        
                        # for monitoring exact memory usage of a single batch inference
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

                    if training_mode == 'SL':
                        finetune_target, finetune_label_weight = find_finetune_flow(te_solution['path_rate'].to(device), data['target_path_rate'].to(device), path_flow_tensor, data['flow_num'])
                        loss_1 = th.sum(loss1_fn(finetune_output, finetune_target.float().to(device)) * finetune_label_weight.to(device))
                        loss = loss_1
                        print("loss:", loss_1)
                    if not (t == T-1 and training_mode == 'SL'):
                        start = time.time()
                        # precalculate network state for solution finetuning process
                        link_capacities = th.tensor(data['link_capacities']).to(device)
                        dem_rate = data['dem_rate'].to(device)
                        path_flow = data['path_flow']
                        path_dem_rate = data['path_dem_rate'].to(device)
                        path_rate = te_solution['path_rate'].to(device)
                        path_weight = data['path_weight'].to(device)
                        path_link_tensor = data['path_link_tensor'].to(device)
                        path_link = data['path_link']
                        print("state precaldulation time:", time.time()-start)
                        
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
                        
                        
                        # solution finetuning process 
                        # We split the args and sample indeces into 2 lists to avoid the heavy memory passing time of the multiprocessing
                        args_list = []
                        args_list2 = [] # record the index of the sample
                        sample_infos = []
                        for k in range(K):    
                            bs_num = int(data['node_num']*alpha)
                            finetune_flow_mask = th.zeros_like(finetune_output).to(device) 
                            sorted_finetune_flow_ind = th.sort(finetune_output, descending=True)[1]
                            finetune_flow_mask[sorted_finetune_flow_ind[:bs_num]] = 1
                            finetune_flow_mask[sorted_finetune_flow_ind[(k+1)*bs_num:(k+2)*bs_num]] = 1
                        
                            
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
                            
                            sample_infos.append((path_selected, finetune_flow_mask)) # for best solution USL loss
                            args = (objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight) 
                            args_list.append(args)
                            args_list2.append((k,))
                        
                        if num_sample_process > 0:
                            with mp.Pool(processes=num_sample_process) as pool:
                                results = pool.map(finetune_fn, args_list2)
                        else:
                            results = []
                            for i in range(K):
                                results.append(finetune_fn(args_list2[i]))
                        print("finish multiple finetuning process time:", time.time()-start)
                        
                        # solution update
                        best_obj = None
                        total_finetuned_flow_indices = set() # set
                        best_finetuned_flow_indices = []
                        best_path_rates = data['init_path_rate']
                        avg_finetune_val = 0
                        sum_log_prob = 0
                        loss_2 = 0
                        for id,ret in enumerate(results):
                            finetune_val, finetune_path_rates, finetuned_flow_indices = ret
                            total_finetuned_flow_indices = total_finetuned_flow_indices.union(set(finetuned_flow_indices))
                            print("id:", id, "finetune_val:", finetune_val)
                        
                        # using total finetuned flow indices for final finetune
                        with th.no_grad():
                            finetune_flow_mask = th.zeros_like(finetune_output).to(device) 
                            total_finetuned_flow_indices = list(total_finetuned_flow_indices)
                            finetune_flow_mask[total_finetuned_flow_indices] = 1
                            
                            path_mask = finetune_flow_mask[path_flow_tensor] # bottleneck
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
                            
                            final_path_selected = path_selected
                            args = (objective, finetune_func, dem_rate_list, path_flow_selected, path_link_selected, path_rate_selected, path_weight_selected, link_usage_background, background_traffic, background_traffic_weight)
                            args_list[0] = args
                            
                            final_finetune_val, final_finetune_path_rates, final_finetuned_flow_indices = finetune_fn((0,))
                            print("final finetune val:", final_finetune_val)
                            print("final finetuned flow indices:", len(final_finetuned_flow_indices), final_finetuned_flow_indices)
                        
                        # when infeasible for numerical issue of gurobi
                        if final_finetune_val == None: 
                            continue 
                        
                        if training_mode == 'USL':
                            # best_finetuned_flow_indices <- total_finetuned_flow_indices
                            finetuned_flow_num = len(final_finetuned_flow_indices)
                            finetune_label_weight = th.ones(flow_num).to(device) / max(flow_num-finetuned_flow_num, 1)
                            finetune_label_weight[final_finetuned_flow_indices] = 1 / max(finetuned_flow_num, 1)
                            best_finetuned_flow = th.zeros(flow_num).to(device)
                            best_finetuned_flow[final_finetuned_flow_indices] = 1
                            loss_1 = th.sum(loss1_fn(finetune_output, best_finetuned_flow) * finetune_label_weight)
                            print("loss:", loss_1)
                            loss = loss_1
                        end = time.time()
                        print("solve finetuning problem time (s):", end-start)

                        print("best_finetune_objval:", best_obj)
                        if training_mode == 'SL':
                            print("opt val:", data['optval'])
                        
                        # update TE solution                        
                        te_solution['path_rate'][final_path_selected] = th.tensor(final_finetune_path_rates)

                        # calculate objective for finetuned TE solution, only for debuging
                        with th.no_grad():
                            path_rate = te_solution['path_rate'].to(device)
                            finetune_mlu = calc_mlu(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                            finetune_avg_weight = calc_avg_weight(path_rate, path_dem_rate, path_weight)
                            finetune_thrpt = calc_throughput(path_rate, path_dem_rate)
                            finetune_traffic_overload = calc_traffic_overload(path_rate, path_dem_rate, path_link_tensor, link_capacities)
                        
                        print("iteration step:", t, "finetune mlu:", finetune_mlu, "finetune avg_weight:", finetune_avg_weight, "finetune thrpt:", finetune_thrpt, "finetune traffic_overload:", finetune_traffic_overload)
                
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train.append(loss.item())


    

    cleanup_dir("./models/%s" % (log_dir))
    cleanup_dir("./log/%s" % (log_dir))
    th.save(model.state_dict(), "./models/%s/model.th" % (log_dir))
    
    with open("./log/%s/training.log" % (log_dir), 'w') as f:
        print("loss_train:", loss_train, file=f)
    # plt.plot(loss_train)
    # plt.savefig("./log/%s/loss_train.png" % (log_dir))
    # plt.clf()


    