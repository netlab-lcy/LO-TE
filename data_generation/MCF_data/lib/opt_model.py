from gurobipy import *
import copy
import numpy as np
import time

'''
nodeNum: int
linkNum: int
demNum: int, indicate the number of flows, for a traffix you may have (n*(n-1)) flows
demands: list, flows' source and destination, each flow represented as a tuple
rates: list, show the flow demand (in traffic matrix)
linkSet: list, each comment represented as a tuple, i.e., (u, v, weight, capacity)
wMatrix and MAXWEIGHT: wMatrix[i][j] < MAXWEIGHT indicates that i,j is a legal link, otherwise, there no link between i,j
mode: 0: bi-directional link share the capacity; 1: bi-directional link do not share the capacity
background_utilities: link utilization ratio at each link from background traffic
env: gurobi environment, for multi-processing
'''
def mcfsolver(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, mode = 1, background_utilities=None, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]
    if background_utilities == None:
        background_utilities = [0.] * linkNum * 2


    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")


    # max link utilization constraints
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        if mode == 1:
            # when link failed, the link capacity is 0 and if and only if there is no flow going through the link, the constraints is achieved 
            model.addConstr(sum1 + background_utilities[h * 2] * linkSet[h][3] <= phi*linkSet[h][3])
            model.addConstr(sum2 + background_utilities[h * 2 + 1] * linkSet[h][3] <= phi*linkSet[h][3])
        else:
            model.addConstr(sum1 + sum2 + background_utilities[h * 2] + background_utilities[h * 2 + 1] <= phi*linkSet[h][3])

    # print("add conservation constraints")
    sumpass = 0
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == inflow[k][j])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin + inflow[k][j] == 0)
            else:
                model.addConstr(sumin + inflow[k][j] == sumout)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        paths = []
        for k in range(demNum):
            path = []
            tmp = []
            for i in range(linkNum):
                util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                path.append(flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X)/rSet[k])
                path.append(flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X)/rSet[k])
                tmp.append(util1)  
                tmp.append(util2)
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
            paths.append(path)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
        paths = []
    
    return optVal, utilities, demand_utilities, paths

'''
pathSet: [[path1_link, path2_link,...], ]
'''
def path_mcfsolver(nodeNum, linkNum, demNum, demands, dem_rates, pathSet, linkSet, dp_ratio=1., env=None):
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rates[i])
    
    split_flow_num = max(1, int(demNum * dp_ratio))
    dp_threshold = np.partition(dem_rates, -split_flow_num)[-split_flow_num]
    
    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    dp_count = 0
    for i in range(demNum):
        path_rate = []
        # support demand pinning
        if rSet[i] >= dp_threshold:
            dp_count += 1
            for j in range(len(pathSet[i])):
                path_rate.append(model.addVar(0, 1, 0, GRB.CONTINUOUS, "f"))
        else:
            print("dp_triggered!")
            path_rate = [0.] * len(pathSet[i])
            path_rate[0] = 1.
        path_rates.append(path_rate)
    
    # max link utilization    
    phi = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "phi")

    link_usage = [0.] * linkNum * 2

    # max link utilization constraints
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    # TODO: avoid bool constraints
    for i in range(linkNum):
        model.addConstr(phi*linkSet[i][3] >= link_usage[i*2])
        model.addConstr(phi*linkSet[i][3] >= link_usage[i*2+1])

    # path rate constraint
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        model.addConstr(sum == 1)

    # Objective
    model.setObjective(phi, GRB.MINIMIZE)
    
    # optimizing
    model.optimize()

    # Get solution
    print("model.status:", model.status)
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        final_path_ratios = []
        for k in range(demNum):
            if rSet[k] >= dp_threshold:
                ratio = []
                for i in range(len(path_rates[k])):
                    ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
                final_path_ratios.append(ratio)
            else:
                final_path_ratios.append(path_rates[k])
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        final_path_ratios = []
    
    return optVal, final_path_ratios


def path_mcfsolver_minpathdiff(nodeNum, linkNum, demNum, demands, dem_rates, pathSet, linkSet, path_rate_origin, mlu_ub, env=None):
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rates[i])

    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff"))
    mlu = model.addVar(mlu_ub, mlu_ub, 0, GRB.CONTINUOUS, "mlu")

    # max link utilization constraints
    link_usage = [0] * linkNum * 2
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= mlu*linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= mlu*linkSet[i][3])

    # path rate constraint
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        model.addConstr(sum == 1)

    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            model.addConstr(path_rate_origin[i][j] - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j]  <= flow_diff[i])
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    model.setObjective(total_diff, GRB.MINIMIZE)

    # optimizing
    model.optimize()
    
    if model.status == GRB.Status.OPTIMAL:
        flag = False
        for i in range(demNum):
            diff_val = flow_diff[i].getAttr(GRB.Attr.X)
            if diff_val > 0.01:
                flow_diff[i].lb = 1
            else:
                if diff_val > 1e-4:
                    flag = True
                flow_diff[i].ub = 0
        mlu.ub = GRB.INFINITY
        mlu.lb = 0
        model.setObjective(mlu, GRB.MINIMIZE)
        model.optimize()
    else:
        print("failed in first solving step.")
    print("Rounding triggered:", flag)   
    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        final_path_ratios = []
        ret_flow_diff = []
    
    return optVal, final_path_ratios, ret_flow_diff


def path_mcfsolver_minweight(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, wMatrix, MAXWEIGHT, mode = 1, env=None):
    sSet = []
    tSet = []
    rSet = []
    total_demand = 0
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])
        total_demand += dem_rate[i]


    # Create optimization model
    model = Model('Discrete_netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)



    # max link utilization constraints
    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= linkSet[i][3])

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1)

    weight_sum = 0
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            path_weight = 0
            for j in pathSet[k][i]:
                path_weight += linkSet[j//2][2]
            weight_sum += path_rates[k][i] * path_weight * rSet[k] / total_demand
    
    # Objective
    model.setObjectiveN(-total_thrpt, index=0, priority=2)
    model.setObjectiveN(weight_sum, index=1, priority=1)
    
    print("start optimizing!")
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        opt_thrpt = -model.getObjective(0).getValue()
        print("thrpt_satisfied ratio:", opt_thrpt/total_demand)
        opt_weight = model.getObjective(1).getValue()
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        opt_thrpt = -1
        opt_weight = -1 
        final_path_ratios = []
    
    return opt_thrpt, opt_weight, final_path_ratios


'''
nodeNum: int
linkNum: int
demNum: int, indicate the number of flows, for a traffix you may have (n*(n-1)) flows
demands: list, flows' source and destination, each flow represented as a tuple
rates: list, show the flow demand (in traffic matrix)
linkSet: list, each comment represented as a tuple, i.e., (u, v, weight, capacity)
wMatrix and MAXWEIGHT: wMatrix[i][j] < MAXWEIGHT indicates that i,j is a legal link, otherwise, there no link between i,j
mode: 0: bi-directional link share the capacity; 1: bi-directional link do not share the capacity
env: gurobi environment, for multi-processing
'''
def mcfsolver_throughput(nodeNum, linkNum, demNum, demands, rates, linkSet, wMatrix, MAXWEIGHT, mode = 1, phi=1.0, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(rates[i])
        src = demands[i][0]
        dst = demands[i][1]
        inflow[i][src] += rSet[i]
        inflow[i][dst] -= rSet[i]


    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    flowVarNum = demNum * linkNum * 2
    flowVarID = 0
    Maps = {}

    for k in range(demNum):
        for i in range(linkNum):
            Maps[(k, (linkSet[i][0], linkSet[i][1]))] = flowVarID
            flowVarID += 1
            Maps[(k, (linkSet[i][1], linkSet[i][0]))] = flowVarID
            flowVarID += 1

    flow = []
    for i in range(flowVarNum):
        flow.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
    
    demand_thrpt = []
    for i in range(demNum):
        demand_thrpt.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "thrpt"))
    

    # max link utilization constraints
    for h in range(linkNum):
        i = linkSet[h][0]
        j = linkSet[h][1]
        sum1 = 0
        sum2 = 0
        for k in range(demNum):
            sum1 += flow[Maps[(k,(i,j))]]
            sum2 += flow[Maps[(k,(j,i))]]
        if mode == 1:
            # when link failed, the link capacity is 0 and if and only if there is no flow going through the link, the constraints is achieved 
            model.addConstr(sum1 <= phi*linkSet[h][3])
            model.addConstr(sum2 <= phi*linkSet[h][3])
        else:
            model.addConstr(sum1 + sum2 <= phi*linkSet[h][3])

    # print("add conservation constraints")
    sumin = 0
    sumout = 0
    for k in range(demNum):
        for j in range(nodeNum):
            sumin = 0
            sumout = 0
            for i in range(nodeNum):
                if wMatrix[i][j] < MAXWEIGHT and i != j:
                    sumin += flow[Maps[(k,(i,j))]]
                    sumout += flow[Maps[(k,(j,i))]]
            if j == demands[k][0]:
                model.addConstr(sumin == 0)
                model.addConstr(sumout == demand_thrpt[k])
            elif j == demands[k][1]:
                model.addConstr(sumout == 0)
                model.addConstr(sumin == demand_thrpt[k])
            else:
                model.addConstr(sumin == sumout)
    
    for i in range(demNum):
        model.addConstr(demand_thrpt[i] <= rSet[i])
    total_thrpt = 0
    for i in range(demNum):
        total_thrpt += demand_thrpt[i]
    # Objective
    model.setObjective(total_thrpt, GRB.MAXIMIZE)
    
    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        # obtain representation vector, deprecated to save time
        
        utilities = [0.] * linkNum * 2
        demand_utilities = []
        paths = []
        for k in range(demNum):
            path = []
            tmp = []
            for i in range(linkNum):
                util1 = flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                util2 = flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X) / (linkSet[i][3] + 1e-5)
                path.append(flow[Maps[(k, (linkSet[i][0], linkSet[i][1]))]].getAttr(GRB.Attr.X)/rSet[k])
                path.append(flow[Maps[(k, (linkSet[i][1], linkSet[i][0]))]].getAttr(GRB.Attr.X)/rSet[k])
                tmp.append(util1)  
                tmp.append(util2)
                utilities[i * 2] += util1 
                utilities[i * 2 + 1] += util2
            demand_utilities.append(tmp)
            paths.append(path)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        utilities = []
        demand_utilities = []
        paths = []
    
    return optVal, utilities, demand_utilities, paths



def path_mcfsolver_throughput(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, phi=1.0, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])


    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)

    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)

    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= phi*linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= phi*linkSet[i][3])

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1)
    
    
    # Objective
    model.setObjective(total_thrpt, GRB.MAXIMIZE)

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        final_path_ratios = []
    
    return optVal, final_path_ratios


def path_mcfsolver_minpathdiff_throughput(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, path_rate_origin, thrpt_lb, scale_factor_lb=1, scale_factor_ub=1, phi=1.0, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])


    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)
    
    start = time.time()
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(0, 1, 0, GRB.CONTINUOUS, "f_diff"))
    scale_factor = model.addVar(scale_factor_lb, scale_factor_ub, 0, GRB.CONTINUOUS, "f_diff")

    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    link_capa_constrs = []
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        link_capa_constrs.append(model.addConstr(link_usage[i*2] <= phi*linkSet[i][3]))
        link_capa_constrs.append(model.addConstr(link_usage[i*2+1] <= phi*linkSet[i][3]))

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1+1e-6) # incase there exists accuracy error in initial path rate
        
    # throughput lb constraint
    c_thrpt = model.addConstr(total_thrpt >= thrpt_lb)
    
    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            model.addConstr(path_rate_origin[i][j] * scale_factor - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j] * scale_factor <= flow_diff[i])
    
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    
    model.setObjective(total_diff, GRB.MINIMIZE)

    # optimizing
    model.optimize()
    print("1st step solving time:", time.time()-start)
    
    # reoptimize with rounded finetune flow decision
    print("Start 2nd step solving.")
    flag = False
    start = time.time()
    if model.status == GRB.Status.OPTIMAL:
        for i in range(demNum):
            diff_val = flow_diff[i].getAttr(GRB.Attr.X)
            
            if diff_val > 0.01:
                flow_diff[i].lb = 1
            else:
                # print("diff_val:", diff_val)
                if diff_val > 1e-4:
                    flag = True
                flow_diff[i].ub = 0
            
            
        model.remove(c_thrpt)
        for constr in link_capa_constrs:
            model.remove(constr)
        
        total_link_overload = 0
        link_overload_vars = []
        for i in range(linkNum):
            link_overload_var = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(link_overload_var >= link_usage[i*2] - phi*linkSet[i][3])
            link_overload_vars.append(link_overload_var)
            total_link_overload += link_overload_var
            link_overload_var = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(link_overload_var >= link_usage[i*2+1] - phi*linkSet[i][3])
            link_overload_vars.append(link_overload_var)
            total_link_overload += link_overload_var
        # scale_factor.lb = 0
        model.update() # Necessary after removing constraints
        model.setObjective(-100 * total_link_overload + total_thrpt, GRB.MAXIMIZE)
        model.optimize()
    else:
        print("failed in first solving step.")
    print("2nd step solving time:", time.time()-start)
    print("Rounding triggered:", flag)  
    
    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        scale_factor_val = scale_factor.getAttr(GRB.Attr.X)

        link_overload_vals = []
        for i in range(linkNum * 2):
            link_overload_vals.append(link_overload_vars[i].getAttr(GRB.Attr.X))
            
        f_thrpt = 0
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                max_path_link_util = 1
                for j in pathSet[k][i]:
                    max_path_link_util = max(1, 1+link_overload_vars[j].getAttr(GRB.Attr.X)/linkSet[j//2][3])
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X)/max_path_link_util)
                f_thrpt += path_rates[k][i].getAttr(GRB.Attr.X)/max_path_link_util * rSet[k]
            final_path_ratios.append(ratio)
        
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        scale_factor_val = 1
        final_path_ratios = []
        ret_flow_diff = []
    
    return optVal, scale_factor_val, final_path_ratios, ret_flow_diff



def path_mcfsolver_minpathdiff_throughput_optimal(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, path_rate_origin, thrpt_lb, scale_factor_lb=0, scale_factor_ub=1, phi=1.0, env=None):
    inflow = [[0.0]*nodeNum for i in range(demNum)]
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])


    # Create optimization model
    model = Model('netflow-throughput', env=env)
    model.setParam("OutputFlag", 0)

    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(vtype=GRB.BINARY, name="f_diff"))
    scale_factor = model.addVar(scale_factor_lb, scale_factor_ub, 0, GRB.CONTINUOUS, "f_diff")

    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= phi*linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= phi*linkSet[i][3])

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1)
    model.addConstr(total_thrpt >= thrpt_lb)
    
    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            model.addConstr(path_rate_origin[i][j] * scale_factor - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j] * scale_factor <= flow_diff[i])
    
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    model.setObjective(total_diff, GRB.MINIMIZE)
    
    # for large-scale model, presolve may cause numerical issue: return the problem as infeasible 
    # model.Params.Presolve = 0
    # model.Params.TimeLimit = 600 # time limit (s)
    model.Params.MIPGap=0.2 # we allow 1% gap to obtain a fast complete 0.2 MIP gap may cause significant gap in Colt session

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        scale_factor_val = scale_factor.getAttr(GRB.Attr.X)
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        scale_factor_val = 1
        final_path_ratios = []
        ret_flow_diff = []
    
    return optVal, scale_factor_val, final_path_ratios, ret_flow_diff


def path_mcfsolver_minpathdiff_optimal(nodeNum, linkNum, demNum, demands, dem_rates, pathSet, linkSet, path_rate_origin, mlu_ub, env=None):
    sSet = []
    tSet = []
    rSet = []
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rates[i])

    # Create optimization model
    model = Model('netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(vtype=GRB.BINARY, name="f_diff"))


    # max link utilization constraints
    link_usage = [0] * linkNum * 2
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= mlu_ub*linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= mlu_ub*linkSet[i][3])

    # path rate constraint
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        model.addConstr(sum == 1)

    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            model.addConstr(path_rate_origin[i][j] - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j]  <= flow_diff[i])
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    model.setObjective(total_diff, GRB.MINIMIZE)
    
    # for large-scale model, presolve may cause numerical issue: return the problem as infeasible 
    # model.Params.Presolve = 0
    # model.Params.TimeLimit = 600 # time limit (s)
    model.Params.MIPGap=0.2 # we allow 5% gap to obtain a fast complete

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        final_path_ratios = []
        ret_flow_diff = []
    
    return optVal, final_path_ratios, ret_flow_diff


def path_mcfsolver_minpathdiff_minweight(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, path_rate_origin, thrpt_lb, weight_ub, wMatrix, MAXWEIGHT, mode = 1, env=None):
    sSet = []
    tSet = []
    rSet = []
    total_demand = 0
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])
        total_demand += dem_rate[i]

    # Create optimization model
    model = Model('Discrete_netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff"))


    # max link utilization constraints
    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    link_capa_constrs = []
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        link_capa_constrs.append(model.addConstr(link_usage[i*2] <= linkSet[i][3]))
        link_capa_constrs.append(model.addConstr(link_usage[i*2+1] <= linkSet[i][3]))

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1+1e-6) # incase there exists accuracy error in initial path rate
    c_constr = model.addConstr(total_thrpt >= thrpt_lb)
            
    # pr gap constraint
    weight_sum = 0
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            path_weight = 0
            for j in pathSet[k][i]:
                path_weight += linkSet[j//2][2]
            weight_sum += path_rates[k][i] * path_weight * rSet[k] / total_demand
    w_constr = model.addConstr(weight_sum <= weight_ub)
    
    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            path_rates[i][j].start = path_rate_origin[i][j]
            model.addConstr(path_rate_origin[i][j] - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j] <= flow_diff[i])
    
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    model.setObjective(total_diff, GRB.MINIMIZE)

    print("start optimizing!")
    model.optimize()
    
    # reoptimize with rounded finetune flow decision
    print("Start 2nd step solving.")
    flag = False
    start = time.time()
    if model.status == GRB.Status.OPTIMAL:
        for i in range(demNum):
            diff_val = flow_diff[i].getAttr(GRB.Attr.X)
            if diff_val > 0.01:
                flow_diff[i].lb = 1
            else:
                if diff_val > 1e-4:
                    flag = True
                flow_diff[i].ub = 0
        model.remove(w_constr)
        model.remove(c_constr)
        for constr in link_capa_constrs:
            model.remove(constr)
        
        total_link_overload = 0
        link_overload_vars = []
        for i in range(linkNum):
            link_overload_var = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(link_overload_var >= link_usage[i*2] - linkSet[i][3])
            link_overload_vars.append(link_overload_var)
            total_link_overload += link_overload_var
            link_overload_var = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f_diff")
            model.addConstr(link_overload_var >= link_usage[i*2+1] - linkSet[i][3])
            link_overload_vars.append(link_overload_var)
            total_link_overload += link_overload_var
            
        model.update() # Necessary after removing constraints
        model.setObjectiveN(weight_sum, index=0, priority=1)
        model.setObjectiveN(-total_thrpt, index=1, priority=2)
        model.setObjectiveN(total_link_overload, index=2, priority=3)
        model.optimize()
    else:
        print("failed in first solving step.")
    print("2nd step solving time:", time.time()-start)
    print("Rounding triggered:", flag)  

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        opt_weight =  model.getObjective(0).getValue()
        opt_thrpt = -model.getObjective(1).getValue()
        opt_total_overload = model.getObjective(2).getValue()
        link_overload_vals = []
        for i in range(linkNum * 2):
            link_overload_vals.append(link_overload_vars[i].getAttr(GRB.Attr.X))
        print("link overload:", max(link_overload_vals) > 1)
            
        f_thrpt = 0
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                max_path_link_util = 1
                for j in pathSet[k][i]:
                    max_path_link_util = max(1, 1+link_overload_vars[j].getAttr(GRB.Attr.X)/linkSet[j//2][3])
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X)/max_path_link_util)
                f_thrpt += path_rates[k][i].getAttr(GRB.Attr.X)/max_path_link_util * rSet[k]
            final_path_ratios.append(ratio)
        print("weight_ub:", weight_ub, "avg_weight:", opt_weight, "satisfied:", opt_weight<=weight_ub, "thrpt:", opt_thrpt, "thrpt_lb:", thrpt_lb, "satisfied:", opt_thrpt>=thrpt_lb)
        print("throughput satisfied:", f_thrpt/total_demand)
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        opt_weight = -1
        final_path_ratios = []
        ret_flow_diff = []
    
    return opt_weight, final_path_ratios, ret_flow_diff


def path_mcfsolver_minpathdiff_minweight_optimal(nodeNum, linkNum, demNum, demands, dem_rate, pathSet, linkSet, path_rate_origin, thrpt_lb, weight_ub, wMatrix, MAXWEIGHT, mode = 1, env=None):
    sSet = []
    tSet = []
    rSet = []
    total_demand = 0
    for i in range(demNum):
        sSet.append(demands[i][0])
        tSet.append(demands[i][1])
        rSet.append(dem_rate[i])
        total_demand += dem_rate[i]


    # Create optimization model
    model = Model('Discrete_netflow', env=env)
    model.setParam("OutputFlag", 0)
    # Create variables
    path_rates = []
    for i in range(demNum):
        path_rate = []
        for j in range(len(pathSet[i])):
            path_rate.append(model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "f"))
        path_rates.append(path_rate)
    flow_diff = []
    for i in range(demNum):
        flow_diff.append(model.addVar(vtype=GRB.BINARY, name="f_diff"))


    # max link utilization constraints
    link_usage = [0] * linkNum * 2

    # max link utilization constraints
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            for j in pathSet[k][i]:
                link_usage[j] += path_rates[k][i] * rSet[k]
    for i in range(linkNum):
        model.addConstr(link_usage[i*2] <= linkSet[i][3])
        model.addConstr(link_usage[i*2+1] <= linkSet[i][3])

    # path rate constraint
    total_thrpt = 0
    for i in range(demNum):
        sum = 0.
        for j in range(len(path_rates[i])):
            sum += path_rates[i][j]
        total_thrpt += sum * rSet[i]
        model.addConstr(sum <= 1)
    model.addConstr(total_thrpt >= thrpt_lb)
            
    # pr gap constraint
    weight_sum = 0
    for k in range(demNum):
        for i in range(len(pathSet[k])):
            path_weight = 0
            for j in pathSet[k][i]:
                path_weight += linkSet[j//2][2]
            weight_sum += path_rates[k][i] * path_weight * rSet[k] / total_demand
    model.addConstr(weight_sum <= weight_ub)
    
    # path diff constraint
    for i in range(demNum):
        for j in range(len(pathSet[i])):
            model.addConstr(path_rate_origin[i][j] - path_rates[i][j] <= flow_diff[i])
            model.addConstr(path_rates[i][j] - path_rate_origin[i][j] <= flow_diff[i])
    # Objective
    total_diff = 0
    for i in range(demNum):
        total_diff += flow_diff[i]
    model.setObjective(total_diff, GRB.MINIMIZE)
    
    # for large-scale model, presolve may cause numerical issue: return the problem as infeasible 
    # model.Params.Presolve = 0
    # model.Params.TimeLimit = 600 # time limit (s)
    model.Params.MIPGap=0.2 # we allow 5% gap to obtain a fast complete

    # optimizing
    model.optimize()

    # Get solution
    if model.status == GRB.Status.OPTIMAL:
        optVal = model.objVal
        optVal = model.objVal
        final_path_ratios = []
        for k in range(demNum):
            ratio = []
            for i in range(len(path_rates[k])):
                ratio.append(path_rates[k][i].getAttr(GRB.Attr.X))
            final_path_ratios.append(ratio)
        ret_flow_diff = []
        for i in range(demNum):
            ret_flow_diff.append(flow_diff[i].getAttr(GRB.Attr.X))
    else:
        print("\n!!!!!!!!!!! No solution !!!!!!!!!!!!!!\n")
        optVal = -1
        final_path_ratios = []
        ret_flow_diff = []
    
    return optVal, final_path_ratios, ret_flow_diff