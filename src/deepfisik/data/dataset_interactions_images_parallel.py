# Copyright (C) 2026, Jaqaman Lab - UTSouthwestern
#
# This file is part of DeepFISIK.
#
# DeepFISIK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepFISIK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepFISIK.  If not, see <http://www.gnu.org/licenses/>.

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
from graphGeneratorInteractionsUNMultipleGraphs import *

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

def get_AP(data):
    AP = np.asarray(data['Association Probability'])[0]

    return torch.tensor(AP, dtype = torch.float)

def get_DR(data):
    DR = np.asarray(data['Dissociation Rate'])[0]

    return torch.tensor(DR, dtype = torch.float)
def get_LF(data):
    LF = np.asarray(data['Label Fraction'])[0]

    return LF

def process_single_file(args):
    raw_path, idx, r, Imean, labelFraction, processed_dir = args

    df = pd.read_csv(raw_path).reset_index(drop=True)
    sim = df

    graph = GraphGeneratorUNMultipleGraphs(sim, r)

    node_feats = get_node_features(sim)
    edge_feats = get_edge_features(graph, r, Imean)
    edge_index = get_adjacency_info(graph)
    RD = get_RD(graph)
    DC = get_DC(graph)
    AP = get_AP(graph)
    DR = get_DR(graph)

    data = Data(x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_feats,
        DC=DC, RD = RD,
        AP = AP, DR = DR,
        LF = labelFraction
        )

    filename = f'data_{idx}.pt'
    torch.save(data, os.path.join(processed_dir, filename))

def process(raw_paths,mat_path,save_path,numberToProcess,missingNumbers):
    r = 1.5/.081
    Imean = 1000/((2**16)-1)

    mat = scipy.io.loadmat(mat_path)
    labelFractions = mat['LF'].flatten()

    args_list = []
    for raw_path in raw_paths:
        fname = os.path.basename(raw_path)
        idx = int(re.search(r'graph_(\d+)\.csv', fname).group(1))

        labelFraction = labelFractions[idx]

        args_list.append(
            (raw_path, idx, r, Imean, labelFraction, save_path)
        )

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        list(tqdm(executor.map(process_single_file,args_list),total=len(args_list)))

raw_paths = 'Datasets/images/superGraph/5/raw/'
mat_path = 'Datasets/images/superGraph/DC_RD_AP_DR/simParamMATLAB_All_2026_02_11.mat'
save_path = 'Datasets/images/superGraph/5/processed/'

file_names = os.listdir(save_path)

numbers = [extract_number(f) for f in file_names]
numbers = [n for n in numbers if n is not None]

sorted_indices = np.argsort(numbers)
sorted_filenames = [file_names[i] for i in sorted_indices]
numbers_sorted = [numbers[i] for i in sorted_indices]

total_array = np.arange(0, 600)

missing_numbers = np.setdiff1d(total_array, numbers_sorted)

sorted_raw_path = get_sorted_full_paths_numerically(raw_paths)
numberToProcess=599
dataset = process(sorted_raw_path,mat_path,save_path,numberToProcess,missing_numbers)
