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
import re
import numpy as np
import pandas as pd
import sys
sys.path.append('/home2/s438168/python/python-applications/receptors/interKinetics/dataGNN')
sys.path.append('/home2/s438168/python/python-applications/receptors/interKinetics/diffusionPredictionGNN')
sys.path.append('/home2/s438168/python/python-applications/receptors/interKinetics/modelGNN')
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

def subGraph(graph, noMoleculesFrame, boundarySize):

    noMoleculesFrame = noMoleculesFrame

    newArea = noMoleculesFrame/graph['Receptor Density'][0]

    newLength = np.sqrt(newArea)/2

    newGraph = graph[(graph['X'] >= boundarySize/2 - newLength) & (graph['X'] <= boundarySize/2 + newLength) & (graph['Y'] >= boundarySize/2 - newLength) & (graph['Y'] <= boundarySize/2 + newLength)].reset_index(drop=True)

    newGraph['noMoleculesFrame'] = noMoleculesFrame

    newGraph['patchSize'] = newLength*2

    return newGraph

def process_single_file(args_list):
    raw_path, idx, save_path = args_list

    graph = pd.read_csv(raw_path)

    newGraph = newGraph = subGraph(graph, np.random.randint(50,250), 30)

    newGraph.to_csv(save_path+'graph_'+str(idx)+'.csv')

def process(raw_paths,save_path,numberToProcess):

        args_list = []
        for idx, raw_path in enumerate(raw_paths):
            args_list.append((raw_path, idx,save_path))
        args_list = args_list[0:numberToProcess]

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            list(tqdm(executor.map(process_single_file,args_list),total=len(args_list)))

raw_paths = '/work/biophysics/s438168/GNN/data/datasetInteractionsTrimers_20250902/archived/'

save_path = '/work/biophysics/s438168/GNN/data/datasetInteractionsTrimers_20250902/raw/'

sorted_raw_path = get_sorted_full_paths_numerically(raw_paths)
numberToProcess=30000
dataset = process(sorted_raw_path,save_path,numberToProcess)
