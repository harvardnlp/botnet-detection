import sys, os
import pandas as pd
import subprocess
from multiprocessing import Pool
import datetime
import csv
import random
import h5py
import numpy as np
import ipaddress
import pathlib
import datetime as datatime
from itertools import product

def write_single_graph(f, graph_id, x, edge_index, y, attrs=None, **kwargs):
    '''
    store into hdf5 file
    '''
    f.create_dataset(f'{graph_id}/x', data=x, dtype = 'float32')
    f.create_dataset(f'{graph_id}/edge_index', data=edge_index, dtype = 'int64')
    f.create_dataset(f'{graph_id}/y', data=y, dtype = 'uint8')
    for key in kwargs:
        f.create_dataset(f'{graph_id}/{key}', data=kwargs[key])
    if attrs is not None:
        for key in attrs:
            f[f'{graph_id}'].attrs[key] = attrs[key]
    return None

def ip2int(ip):
    '''
    convert x.x.x.x into a number
    '''
    try:
        ip = ip.split(',')[0]
        ip = ipaddress.ip_address(ip)
        ip = int(ip)
        return ip
    except:
        return random.randint(0, 1<<32)

def search_dict(IP, IP_dict):
    '''
    use a dictionary to renumber the IPs into 0,1,2,...
    '''
    if IP not in IP_dict:
        IP_dict[IP] = len(IP_dict)
    return IP_dict[IP]

def prepare_background(f, dst_dir, dst_name, graph_id, start_time, stop_time):
    '''
    Transform txt files into standard hdf5 format
    arg = [txt_file_name, subgroup of graphs]
    '''

    #read data
    df = pd.read_csv(f, sep = '@')#, nrows = 10000)#
    df.columns = ["time", "srcIP", "dstIP"]

    #filter time
    start_time_formated = datetime.datetime.strptime(start_time, "%Y%m%d%H%M%S")
    stop_time_formated = datetime.datetime.strptime(stop_time, "%Y%m%d%H%M%S")
    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x[:21], "%b %d, %Y %H:%M:%S"))
    df = df[ df.time >= start_time_formated]
    df = df[ df.time < stop_time_formated]

    #transform time and IP address into formal type
    df["srcIP"] = df["srcIP"].apply(ip2int)
    df["dstIP"] = df["dstIP"].apply(ip2int)
    
    #aggregate nodes, build dictionary
    df['srcIP'] = df['srcIP'].apply(lambda x: x >> 8)#
    df['dstIP'] = df['dstIP'].apply(lambda x: x >> 8)#
    df = df.drop_duplicates()
    
    #renumber into 0, 1, 2, ..
    IP_dict = {}
    df["srcIP"] = df["srcIP"].apply(lambda x : search_dict(x, IP_dict))
    df["dstIP"] = df["dstIP"].apply(lambda x : search_dict(x, IP_dict))

    #write into h5py files
    num_nodes = len(IP_dict)
    num_edges = df.shape[0]
    
    f_h5py = h5py.File(os.path.join(dst_dir,dst_name), 'a')
    write_single_graph(f_h5py, 
                        graph_id = graph_id, 
                        x = np.ones([num_nodes, 1]), 
                        edge_index = np.array(df[["srcIP", "dstIP"]]).T,
                        y = np.zeros(num_nodes), 
                        attrs={'num_nodes': num_nodes, 'num_edges': num_edges, 'num_evils':0})
    f_h5py.close()

if __name__ == '__main__':
    prepare_background('equinix-nyc.dirA.20181220-131256.UTC.anon.pcap', '.', 'tmp.hdf5', 0, '20181220081256', '20181220081257')
