import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import scipy.io
from deepfisik.data.graph_generator_interactions_trimers import *

import re

def get_sorted_full_paths_numerically(directory_path):
    """
    Returns a list of full paths of files in a directory,
    sorted numerically based on a numeric part in their names.
    """
    full_paths = []
    for item in os.listdir(directory_path):
        full_path = os.path.join(directory_path, item)
        if os.path.isfile(full_path):  # Ensure it's a file
            full_paths.append(full_path)

    def extract_numeric_key(path):
        filename = os.path.basename(path)
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group(0))
        return float('inf') # Place items without numbers at the end

    return sorted(full_paths, key=extract_numeric_key)

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def get_node_features(data):
    data = data.dropna().reset_index(drop=True)

    all_node_features = []
    X = []
    Y = []
    intensity = []

    X.append(data['X'].to_numpy())
    X = np.concatenate(X).ravel()

    Y.append(data['Y'].to_numpy())
    Y = np.concatenate(Y).ravel()

    intensity.append(data['Intensity'].to_numpy())
    intensity = np.concatenate(intensity).ravel()

    intensity = (intensity-intensity.min())/(intensity.max()-intensity.min())

    all_node_features.append(X)
    all_node_features.append(Y)
    all_node_features.append(intensity.astype(float))

    all_node_features = np.transpose(np.asarray(all_node_features))

    return torch.tensor(all_node_features, dtype=torch.float)

def get_edge_features(data,r,Imean):
    all_edge_features = []
    edge_distance_prob = []

    edge_distance_prob.append(data['edge_distance']/r)
    edge_distance_prob = np.asarray(edge_distance_prob).flatten()

    kFactor= 3

    intensity_difference_prob_0=np.asarray(np.exp((-kFactor*(data['intensity_difference']**2))/(Imean**2)))

    intensity_difference_prob_p1=np.asarray(np.exp((-kFactor*((data['intensity_difference']-Imean)**2))/(Imean**2)))
    intensity_difference_prob_p2=np.asarray(np.exp((-kFactor*((data['intensity_difference']-2*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p3=np.asarray(np.exp((-kFactor*((data['intensity_difference']-3*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p4=np.asarray(np.exp((-kFactor*((data['intensity_difference']-4*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p5=np.asarray(np.exp((-kFactor*((data['intensity_difference']-5*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p6=np.asarray(np.exp((-kFactor*((data['intensity_difference']-6*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p7=np.asarray(np.exp((-kFactor*((data['intensity_difference']-7*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p8=np.asarray(np.exp((-kFactor*((data['intensity_difference']-8*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p9=np.asarray(np.exp((-kFactor*((data['intensity_difference']-9*Imean)**2))/(Imean**2)))
    intensity_difference_prob_p10=np.asarray(np.exp((-kFactor*((data['intensity_difference']-10*Imean)**2))/(Imean**2)))

    intensity_difference_prob_n1=np.asarray(np.exp((-kFactor*((data['intensity_difference']+Imean)**2))/(Imean**2)))
    intensity_difference_prob_n2=np.asarray(np.exp((-kFactor*((data['intensity_difference']+2*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n3=np.asarray(np.exp((-kFactor*((data['intensity_difference']+3*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n4=np.asarray(np.exp((-kFactor*((data['intensity_difference']+4*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n5=np.asarray(np.exp((-kFactor*((data['intensity_difference']+5*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n6=np.asarray(np.exp((-kFactor*((data['intensity_difference']+6*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n7=np.asarray(np.exp((-kFactor*((data['intensity_difference']+7*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n8=np.asarray(np.exp((-kFactor*((data['intensity_difference']+8*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n9=np.asarray(np.exp((-kFactor*((data['intensity_difference']+9*Imean)**2))/(Imean**2)))
    intensity_difference_prob_n10=np.asarray(np.exp((-kFactor*((data['intensity_difference']+10*Imean)**2))/(Imean**2)))

    intensity_difference_prob_p = np.max([intensity_difference_prob_p1,intensity_difference_prob_p2,intensity_difference_prob_p3,
    intensity_difference_prob_p4,intensity_difference_prob_p5,intensity_difference_prob_p6,
    intensity_difference_prob_p7,intensity_difference_prob_p8,intensity_difference_prob_p9,
    intensity_difference_prob_p10],axis=0)

    intensity_difference_prob_n = np.max([intensity_difference_prob_n1,intensity_difference_prob_n2,intensity_difference_prob_n3,
    intensity_difference_prob_n4,intensity_difference_prob_n5,intensity_difference_prob_n6,
    intensity_difference_prob_n7,intensity_difference_prob_n8,intensity_difference_prob_n9,
    intensity_difference_prob_n10],axis=0)

    all_edge_features.append(edge_distance_prob)

    all_edge_features.append(intensity_difference_prob_n)

    all_edge_features.append(intensity_difference_prob_0)

    all_edge_features.append(intensity_difference_prob_p)
    all_edge_features = np.transpose(np.asarray(all_edge_features))

    return torch.tensor(all_edge_features, dtype=torch.float)

def get_adjacency_info( data):

    edges_forward = []

    edges_forward.append(data['objectID_N'])
    edges_forward.append(data['objectID_N1'])
    edges_forward = (np.asarray(edges_forward)).astype(int)

    return torch.tensor(edges_forward, dtype=torch.long)

def get_RD(data):
    RD = np.asarray(data['Receptor Density'])[0]

    return torch.tensor(RD, dtype = torch.float)
def get_DC(data):

    label = np.asarray(data['Diffusion Coefficient'])[0]

    return torch.tensor(label, dtype=torch.float)

def get_AP1(data):

    AP1 = np.asarray(data['AP1'])[0]

    return torch.tensor(AP1, dtype = torch.float)

def get_AP2(data):

    AP2 = np.asarray(data['AP2'])[0]

    return torch.tensor(AP2, dtype = torch.float)

def get_DR1(data):

    DR1 = np.asarray(data['DR1'])[0]

    return torch.tensor(DR1, dtype = torch.float)

def get_DR2(data):

    DR2 = np.asarray(data['DR2'])[0]

    return torch.tensor(DR2, dtype = torch.float)

def get_LF(data):
    LF = np.asarray(data['Label Fraction'])[0]

    return LF

def process_single_file(args):
    raw_path, idx, r, Imean, labelFraction, processed_dir = args

    df = pd.read_csv(raw_path).reset_index(drop=True)
    index_movies = df['Movie'].drop_duplicates().index.to_numpy()

    for i, index_movie in enumerate(index_movies):
        if i != len(index_movies) - 1:
            sim = df.iloc[index_movie:index_movies[i + 1]].reset_index(drop=True)
        else:
            sim = df.iloc[index_movie:].reset_index(drop=True)

        graph = GraphGeneratorTrimers(sim, r)

        node_feats = get_node_features(sim)
        edge_feats = get_edge_features(graph, r, Imean)
        edge_index = get_adjacency_info(graph)
        RD = get_RD(graph)
        DC = get_DC(graph)
        AP1 = get_AP1(graph)
        DR1 = get_DR1(graph)
        AP2 = get_AP2(graph)
        DR2 = get_DR2(graph)

        data = Data(x=node_feats,
            edge_index=edge_index,
            edge_attr=edge_feats,
            DC=DC, RD = RD,
            AP1 = AP1, DR1 = DR1,
            AP2 = AP2,DR2 = DR2,
            LF = labelFraction
            )

        filename = f'data_{idx}.pt'
        torch.save(data, os.path.join(processed_dir, filename))

def process(raw_paths,mat_path,save_path,numberToProcess):

    r = 1.5
    Imean = 1

    mat = scipy.io.loadmat(mat_path)
    labelFractions = mat['LF'].flatten()

    args_list = []
    for raw_path in raw_paths:
        idx = extract_number(os.path.basename(raw_path))
        labelFraction = labelFractions[idx]
        args_list.append((raw_path, idx, r, Imean, labelFraction, save_path))
    args_list = args_list[0:numberToProcess]

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_file,args_list),total=len(args_list)))

raw_paths = '/project/biophysics/jaqaman_lab/interKinetics/knguyen/GNN/2025/data/dataInteractionsPureSimTrimers_20250724/raw/'
mat_path = '/project/biophysics/jaqaman_lab/interKinetics/knguyen/GNN/2025/data/dataInteractionsPureSimTrimers_20250724/DC_RD_AP_DR/simParamMATLAB_All_2024_07_24.mat'
save_path = '/project/biophysics/jaqaman_lab/interKinetics/knguyen/GNN/2025/data/dataInteractionsPureSimTrimers_20250724/processed/'

sorted_raw_path = get_sorted_full_paths_numerically(raw_paths)
numberToProcess=30000
dataset = process(sorted_raw_path,mat_path,save_path,numberToProcess)
