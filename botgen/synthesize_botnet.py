import numpy as np
import pandas as pd 
import h5py
from background import write_single_graph
import random
import math
from itertools import product
import os, sys
from background import write_single_graph

def chord(num_node, num_edge, interval):
    edge = [[i%num_node, (i+1)%num_node]for i in range(num_node)]
    
    fingers = [x for x in range(0,num_node, interval)]
    for (i, finger) in enumerate(fingers):
        for j in range(i+1, len(fingers)):
            edge.append([finger, fingers[j]])
    
    edge_select = random.sample(edge, num_edge) 
    return np.array(edge_select)

def debru(num_node, num_edge, m): 

    n = int(np.ceil( math.log(num_node, m)))
    nodes = []
    for p in product(range(m), repeat = n):
        nodes.append(p)
    d_dict = {}
    for key in random.sample(nodes, num_node):
        d_dict[key] = len(d_dict)

    node = [key for key in d_dict]
    debru_edge = []
    for output in list(product(node, range(m))):
        if output[0] in d_dict and (output[0][1:] + (output[1],)) in d_dict:
            if d_dict[ output[0]] != d_dict[(output[0][1:] + (output[1],))]:
                debru_edge.append([d_dict[ output[0]], d_dict[(output[0][1:] + (output[1],))]])
    
    edge_select = random.sample(debru_edge, num_edge)
    return np.array(edge_select)

def leet(num_node, num_edge):
    edge = []
    log_n = int(math.log(num_node,2))+1

    for cnt in range(1,log_n):
        edge += [[ start,(start + (1<<cnt))%num_node] for start in range(cnt,num_node,log_n)]
    edge += [[i,(i+1)%num_node] for i in range(num_node)]
    
    edge_select = random.sample(edge, num_edge)
    return np.array(edge_select)

def binarySearch(nodes, item):
    '''
    return the index of the maximum value < item
    '''
    first = 0
    last = len(nodes)-1
    if(item <= nodes[0]):
        return -1
    if(item > nodes[last]):
        return last
    while (last - first > 1):
        midpoint = int((first + last) / 2)
        if(nodes[midpoint] < item and item <= nodes[midpoint+1]):
            first = midpoint
            break
        elif (nodes[midpoint] >= item):
            last = midpoint
        else:
            first = midpoint
    return first

def kadem(n, n_edge, k, bit):
    nodes = random.sample(range(1<<bit), n)

    nodes.sort()
    kademlia_edges = {}
    pr = 1.0*n_edge/(k*bit*n)
    for i in range(n):#n
        node_move = nodes[i]
        for j in range(bit):
            if((node_move >> j)%2 == 1):
                range_lower = ((node_move >> (j+1)) << (j+1))
                range_upper = ((node_move >> (j+1)) << (j+1)) + (1<<j)
            else:
                range_lower = ((node_move >> (j+1)) << (j+1)) + (1<<j)
                range_upper = ((node_move >> (j+1)) << (j+1)) + (1<<(j+1))
            index_range_lower = binarySearch(nodes, range_lower) + 1
            index_range_upper = binarySearch(nodes, range_upper) + 1
            range_list = range(index_range_lower,index_range_upper)

            if(len(range_list) >= k):
                kademlia_edges[(i,j)] = random.sample(range_list,k)
            else:
                kademlia_edges[(i,j)] = range_list
    edge = []
    for x in kademlia_edges:
        edge += [ [x[0], y] for y in kademlia_edges[x]]

    edge_select = random.sample(edge, n_edge)
    return np.array(edge_select)

def write_botnet(dst_dir, dst_name, graph_id, edges):
    f = h5py.File(os.path.join(dst_dir,dst_name), 'a')
    x, y, edge_index = np.array(f[f'{graph_id}/x']), np.array(f[f'{graph_id}/y']), np.array(f[f'{graph_id}/edge_index'])
    
    #select evil node randomly
    evil_edges = np.array(edges).T
    evil_original = list(set(evil_edges[0,:].tolist()+evil_edges[1,:].tolist()))
    num_evil = len(evil_original)
    evil = random.sample(range(x.shape[0]), num_evil) 
    
    evil_dict = {evil_original[i]:evil[i] for i in range(num_evil)}
    for row in range(evil_edges.shape[0]):
        for col in range(evil_edges.shape[1]):
            evil_edges[row, col] = evil_dict[evil_edges[row, col]]
    
    edge_index = np.hstack([edge_index,evil_edges])
    y[evil] = 1

    del f[f'{graph_id}']
    write_single_graph(f, 
                        graph_id = graph_id, 
                        x = x, 
                        edge_index = edge_index,
                        y = y, 
                        attrs={'num_nodes': x.shape[0], 'num_edges': edge_index.shape[1], 'num_evils':num_evil})
    f.close()


if __name__ == '__main__':
    print(kadem(10,10,3,4))