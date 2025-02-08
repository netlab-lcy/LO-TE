from gurobipy import *
import numpy as np
import time


'''
we assume init path rate always satisfy demand ratio constraints (sum <= 1)
precalculate link_util_background to avoid the building up time
we only consider the corresponding paths for the selected flows for finetuning
*Note that the demand number is the original demand number, not the selected demand number
'''
def finetune_path_mcfsolver_throughput(nodeNum, linkNum, demNum, dem_rate, path_link, path_flow, path_rate_origin, link_capacities, link_usage_background, background_traffic, phi=1.0, min_path_diff=False, env=None):
    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)

    # Create variables
    path_num = len(path_link)
    path_rates = []
    for i in range(path_num):
        path_rates.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    
    overload_traffic = []
    total_overload_traffic = 0.
    for i in range(linkNum):
        overload_traffic.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "traffic"))
        total_overload_traffic += overload_traffic[-1]
    
    if min_path_diff:
        flow_diff = {}
        for i in range(path_num):
            if path_flow[i] not in flow_diff:
                flow_diff[path_flow[i]] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(path_rate_origin[i] - path_rates[i] <= flow_diff[path_flow[i]])
            model.addConstr(path_rates[i] - path_rate_origin[i]  <= flow_diff[path_flow[i]])
        total_diff = 0
        for value in flow_diff.values():
            total_diff += value
        model.setObjectiveN(total_diff, index=2, priority=0)
    
    link_usage = [0.] * linkNum
    # max link utilization constraints
    for k in range(path_num):
        for j in path_link[k]:
            link_usage[j] += path_rates[k] * dem_rate[path_flow[k]]
           
    for i in range(linkNum):
        # incase congestion on current te solution
        model.addConstr(overload_traffic[i] >= link_usage_background[i] + link_usage[i] - phi*link_capacities[i])
    
    
    # path rate constraint
    total_thrpt = background_traffic
    demand_thrpt_ratio = [0.] * demNum
    for k in range(path_num):
        demand_thrpt_ratio[path_flow[k]] += path_rates[k]
        total_thrpt += path_rates[k] * dem_rate[path_flow[k]]
    
    for i in range(demNum):
        if type(demand_thrpt_ratio[i]) != float:
            model.addConstr(demand_thrpt_ratio[i] <= 1)
    
    # Objective
    model.setObjectiveN(-total_thrpt, index=0, priority=1)
    model.setObjectiveN(total_overload_traffic, index=1, priority=2)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        opt_thrpt = -model.getObjective(0).getValue()
        opt_overload_traffic = model.getObjective(1).getValue()
        final_path_ratios = []
        for k in range(path_num):
            final_path_ratios.append(path_rates[k].getAttr(GRB.Attr.X))
        if min_path_diff:
            finetuned_flow = []
            for key, value in flow_diff.items():
                if value.getAttr(GRB.Attr.X) > 1e-4:
                    finetuned_flow.append(key)
        else:
            finetuned_flow = None
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        opt_thrpt = None 
        opt_overload_traffic = None
        final_path_ratios = []
        finetuned_flow = []
        
    return (opt_overload_traffic, -opt_thrpt), final_path_ratios, finetuned_flow



'''
we assume init path rate always satisfy demand ratio constraints (sum <= 1)
precalculate link_util_background to avoid the build up time
we only consider the corresponding paths for the selected flows for finetuning
'''
def finetune_path_mcfsolver_mlu(nodeNum, linkNum, demNum, dem_rate, path_link, path_flow, path_rate_origin, link_capacities, link_usage_background, background_traffic, min_path_diff=False, phi=1.0, failed_link=[], env=None):
    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)

    # Create variables
    path_num = len(path_link)
    path_rates = []
    for i in range(path_num):
        path_rates.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))

    if min_path_diff:
        flow_diff = {}
        for i in range(path_num):
            if path_flow[i] not in flow_diff:
                flow_diff[path_flow[i]] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(path_rate_origin[i] - path_rates[i] <= flow_diff[path_flow[i]])
            model.addConstr(path_rates[i] - path_rate_origin[i]  <= flow_diff[path_flow[i]])
        total_diff = 0
        for value in flow_diff.values():
            total_diff += value
        model.setObjectiveN(total_diff, index=1, priority=0)

    mlu = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "mlu")

    link_usage = [0.] * linkNum
    for k in range(path_num):
        for j in path_link[k]:
            link_usage[j] += path_rates[k] * dem_rate[path_flow[k]]
   
    # max link utilization constraints      
    for i in range(linkNum):
        if i in failed_link:
            if isinstance(link_usage_background[i] + link_usage[i] <=0, np.bool_):
                assert link_usage_background[i] + link_usage[i] <= 0
            else:
                model.addConstr(link_usage_background[i] + link_usage[i] <= 0)
        else:
            model.addConstr(mlu >= (link_usage_background[i] + link_usage[i]) / link_capacities[i] )
    
    # path rate constraint
    demand_thrpt_ratio = [0.] * demNum
    for k in range(path_num):
        demand_thrpt_ratio[path_flow[k]] += path_rates[k]
    
    for i in range(demNum):
        if type(demand_thrpt_ratio[i]) != float:
            model.addConstr(demand_thrpt_ratio[i] == 1)
    
    # Objective
    model.setObjectiveN(mlu, index=0, priority=1)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        opt_mlu = model.getObjective(0).getValue()
        final_path_ratios = []
        for k in range(path_num):
            final_path_ratios.append(path_rates[k].getAttr(GRB.Attr.X))
        if min_path_diff:
            finetuned_flow = []
            for key, value in flow_diff.items():
                if value.getAttr(GRB.Attr.X) > 1e-4:
                    finetuned_flow.append(key)
        else:
            finetuned_flow = None
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        opt_mlu = None 
        final_path_ratios = []
        finetuned_flow = []
        
    return (opt_mlu),  final_path_ratios, finetuned_flow


'''
we assume init path rate always satisfy demand ratio constraints (sum <= 1)
precalculate link_util_background to avoid the build up time
we only consider the corresponding paths for the selected flows for finetuning
We precalculate path weight to reduce the model build up time
'''
def finetune_path_mcfsolver_throughput_weight(nodeNum, linkNum, demNum, dem_rate, path_link, path_flow, path_rate_origin, path_weight, link_capacities, link_usage_background, background_traffic, background_traffic_weight, min_path_diff=False, phi=1.0, env=None):
    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)

    # Create variables
    path_num = len(path_link)
    path_rates = []
    for i in range(path_num):
        path_rates.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    
    # min path diff constraints
    if min_path_diff:
        flow_diff = {}
        for i in range(path_num):
            if path_flow[i] not in flow_diff:
                flow_diff[path_flow[i]] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(path_rate_origin[i] - path_rates[i] <= flow_diff[path_flow[i]])
            model.addConstr(path_rates[i] - path_rate_origin[i]  <= flow_diff[path_flow[i]])
        total_diff = 0
        for value in flow_diff.values():
            total_diff += value
        model.setObjectiveN(total_diff, index=3, priority=0)

    # overload traffic constraints
    overload_traffic = []
    total_overload_traffic = 0.
    for i in range(linkNum):
        overload_traffic.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "traffic"))
        total_overload_traffic += overload_traffic[-1]

    link_usage = [0.] * linkNum
    # max link utilization constraints
    for k in range(path_num):
        for j in path_link[k]:
            link_usage[j] += path_rates[k] * dem_rate[path_flow[k]]
    
    weight_sum = 0
    for k in range(path_num):
        weight_sum += path_weight[k] * path_rates[k]
          
    for i in range(linkNum):
        # incase congestion on current te solution
        model.addConstr(overload_traffic[i] >= link_usage_background[i] + link_usage[i] - phi*link_capacities[i])
    
    # path rate constraint
    total_thrpt = background_traffic
    demand_thrpt_ratio = [0.] * demNum
    for k in range(path_num):
        demand_thrpt_ratio[path_flow[k]] += path_rates[k]
        total_thrpt += path_rates[k] * dem_rate[path_flow[k]]
    for i in range(demNum):
        if type(demand_thrpt_ratio[i]) != float:
            model.addConstr(demand_thrpt_ratio[i] <= 1)
    
    # Objective
    model.setObjectiveN(weight_sum, index=0, priority=1)
    model.setObjectiveN(-total_thrpt, index=1, priority=2)
    model.setObjectiveN(total_overload_traffic, index=2, priority=3)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        opt_weight = model.getObjective(0).getValue()
        opt_thrpt = -model.getObjective(1).getValue()
        opt_overload_traffic = model.getObjective(2).getValue()
        final_path_ratios = []
        for k in range(path_num):
            final_path_ratios.append(path_rates[k].getAttr(GRB.Attr.X))
        if min_path_diff:
            finetuned_flow = []
            for key, value in flow_diff.items():
                if value.getAttr(GRB.Attr.X) > 1e-4:
                    finetuned_flow.append(key)
        else:
            finetuned_flow = None
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        opt_weight = None
        opt_thrpt = None 
        opt_overload_traffic = None
        final_path_ratios = []
        finetuned_flow = []
        
    return (opt_overload_traffic, -opt_thrpt, opt_weight+background_traffic_weight),  final_path_ratios, finetuned_flow
