'''
This file download file from CAIDA to the same folder where the .py is
python3 download_pcap.py
'''
import sys,os
import subprocess
from multiprocessing import Pool

def download_and_decompress(raw_link, CAIDAUser, CAIDAPassword):
    '''
    Download public dataset from CAIDA networks
    - raw_link = ='https://data.caida.org/datasets/passive-2018/equinix-nyc/20181220-130000.UTC/equinix-nyc.dirA.20181220-131256.UTC.anon.pcap.gz'
    - Output: pcap files. Example: equinix-nyc.dirA.20181115-130000.UTC.anon.pcap
    '''
    pcap_file = raw_link.split('/')[-1]
    pcap_file_decompressed = pcap_file[:-3]
    
    tmp_file = pcap_file + '.tmp'

    #download the file and name it.tmp
    cmd = f'wget --user {CAIDAUser} --password {CAIDAPassword} {raw_link} -O {tmp_file}' 
    os.system(cmd)

    # Rename file name when download is complete
    os.rename(tmp_file, pcap_file)
    
    #decompress the data
    cmd = f'gunzip {pcap_file}'
    os.system(cmd)   
    
    #read time, src, dst
    cmd = f'tshark -r {pcap_file_decompressed} -Y \"ip\" -T fields -E separator=@ -e frame.time -e ip.src -e ip.dst > {tmp_file}'

    # Rename file name when reading is complete
    cmd = f'mv {tmp_file} {pcap_file_decompressed}'

    return pcap_file_decompressed


