#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
Objective: routing selection idea (TM classification idea)
'''

import numpy as np
import json

'''
tm_range of target TMs, note that for gravity we always load all the file since we normalize the rates with original gravity TM (gravity[0])
'''
class ReadTopo:
    def __init__(self, data_dir, topo_name, TM_info):
        self.__data_dir = data_dir
        self.__topo_name = topo_name
        self.__topofile = data_dir + "/topology/" + topo_name + "_topo.txt"
       
        self.__pathfile = data_dir + "/path_data/" + topo_name + ".json" 

        self.__TM_info = TM_info
        
        # store topo info
        self.__nodenum = 0
        self.__linknum = 0
        self.__linkset = []
        self.__wMatrix = []
        self.__cMatrix = []
        self.__MAXWEIGHT = 99999

        # store self.__demands
        self.__demnum = 0
        self.__demands = []

        # store paths and rates
        self.__path_info = None
        self.__demrates = {}
        
        self.get_topo()
        self.get_demands()
        self.get_paths()
        self.get_demrates()

    def get_topo(self):
        file = open(self.__topofile)
        lines = file.readlines()
        file.close()
        lineList = lines[0].strip().split()
        self.__nodenum = int(lineList[0])
        self.__linknum = int(lineList[1])
        for i in range(self.__nodenum):
            self.__wMatrix.append([])
            self.__cMatrix.append([])
            for j in range(self.__nodenum):
                if i == j:
                    self.__wMatrix[i].append(0)
                else:
                    self.__wMatrix[i].append(self.__MAXWEIGHT)
                self.__cMatrix[i].append(0)

        for i in range(1, self.__linknum+1):
            lineList = lines[i].strip().split()
            left = int(lineList[0]) - 1
            right = int(lineList[1]) - 1
            capa = float(lineList[3])
            weight = int(lineList[2])
            self.__linkset.append([left, right, weight, capa])
            self.__wMatrix[left][right] = weight
            self.__wMatrix[right][left] = weight
            self.__cMatrix[left][right] = capa 
            self.__cMatrix[right][left] = capa

    def get_demands(self):
        self.__demnum = self.__nodenum*(self.__nodenum - 1)
        for src in range(self.__nodenum):
            for dst in range(self.__nodenum):
                if src == dst:
                    continue
                self.__demands.append([src,dst])

    def get_paths(self):
        with open(self.__pathfile, "r") as f:
            line = f.readline()
            self.__path_info = json.loads(line)

    def get_demrates(self):
        for tm_type in self.__TM_info:
            tm_range = self.__TM_info[tm_type]
            ratefile = self.__data_dir + "/traffic/" + tm_type + "/" + self.__topo_name + "_TM.txt"
            file = open(ratefile)
            lines = file.readlines()
            file.close()
            if tm_range is not None:
                lines = lines[tm_range[0]:tm_range[1]]
            demrates = []
            for i in range(len(lines)):
                lineList = lines[i].strip().split(',')
                rates = list(map(float, lineList))
                demrates.append(rates)
            self.__demrates[tm_type] = demrates


   
    def read_info(self):
        return self.__nodenum, self.__linknum, self.__linkset, self.__demnum, self.__demands, self.__demrates, self.__path_info, self.__cMatrix, self.__wMatrix, self.__MAXWEIGHT


