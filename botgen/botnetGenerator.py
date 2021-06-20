import argparse
import os, sys
from download_pcap import download_and_decompress
from background import prepare_background
from synthesize_botnet import *

def parse_args():
    args = argparse.ArgumentParser(description='Generate a botnet.')
    # CAIDA account
    args.add_argument('--CAIDA_user', type=str, help='Username for CAIDA')
    args.add_argument('--CAIDA_password', type=str, help='Password for CAIDA')
    # File src & dst
    args.add_argument('--CAIDA_link', type=str, default='https://data.caida.org/datasets/passive-2018/equinix-nyc/20181220-130000.UTC/equinix-nyc.dirA.20181220-131256.UTC.anon.pcap.gz', help='CAIDA file link')
    args.add_argument('--dst_dir', type=str, default='.', help='Destination Directory')
    args.add_argument('--dst_name', type=str, default='tmp.hdf5', help='Destination filename')
    # Details for background
    args.add_argument('--graph_id', type=int, default='0', help='graph id')
    args.add_argument('--start_time', type=str, default='20181220081256', help='start_time, e.g.20181220081256')
    args.add_argument('--stop_time', type=str, default='20181220081257', help='stop_time, e.g.20181220081257')
    # Details for botnet
    args.add_argument('--botnet_type', type=str, default='chord', help='botne type: chord, debru, kadem, leet')
    args.add_argument('--num_edge', type=int, default=10, help='num of edges in botnet')
    args.add_argument('--num_node', type=int, default=10, help='num of nodes in botnet')
    # chord
    args.add_argument('--interval', type=int, default=2 , help='interval for chord')
    # debru
    args.add_argument('--m', type=int, default=2 , help='m for debru')
    # kadem
    args.add_argument('--k', type=int, default=3 , help='k for kadem')
    args.add_argument('--bit', type=int, default=4 , help='bit for kadem')

    args = args.parse_args()
    return args  

if __name__=="__main__":
    args = parse_args()
    print('name = %s'%args.botnet_type)

    #pcap_file_decompressed = 'equinix-nyc.dirA.20181220-131256.UTC.anon.pcap'
    pcap_file_decompressed = download_and_decompress(args.CAIDA_link, args.CAIDA_user, args.CAIDA_password)
    prepare_background(pcap_file_decompressed, args.dst_dir, args.dst_name, args.graph_id, args.start_time, args.stop_time)
    os.remove(pcap_file_decompressed)

    if args.botnet_type == 'chord':
        edges = chord(args.num_node, args.num_edge, args.interval)
    elif args.botnet_type == 'leet':
        edges = leet(args.num_node, args.num_edge)
    elif args.botnet_type == 'kadem':
        edges = kadem(args.num_node, args.num_edge, args.k, args.bit)
    elif args.botnet_type == 'debru':
        edges = debru(args.num_node, args.num_edge, args.m)

    write_botnet(args.dst_dir, args.dst_name, args.graph_id, edges)


