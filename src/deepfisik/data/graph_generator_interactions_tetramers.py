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

import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def correctIndex(graph):

    last_frame = graph.drop_duplicates(subset=['objectID_N1']).reset_index()
    first_frame = graph.drop_duplicates(subset=['objectID_N']).reset_index()
    finalGraph = graph.copy()

    oID = []
    oID.append(first_frame['objectID_N'].to_numpy())
    oID.append(last_frame['objectID_N1'].to_numpy())
    oID = np.concatenate(oID).ravel()
    oID = oID[np.sort(np.unique(oID, return_index=True)[1])]
    ref = np.arange(0,len(oID),1)
    diff = oID-ref

    if bool(np.any(diff)):
        oIDFirstFrame = oID[0:len(first_frame)]
        diffFirstFrame = diff[0:len(first_frame)]
        oIDLastFrame = oID[len(first_frame):]
        diffLastFrame = diff[len(first_frame):]

        for i in range(len(oID)):
            idxFirstFrame = np.asarray(np.where(np.isin(graph['objectID_N'],oID[i]))).flatten()
            finalGraph.loc[idxFirstFrame, 'objectID_N'] =  finalGraph.loc[idxFirstFrame, 'objectID_N'] - diff[i]

        for i in range(len(oID)):
            idxLastFrame = np.asarray(np.where(np.isin(graph['objectID_N1'],oID[i]))).flatten()
            finalGraph.loc[idxLastFrame, 'objectID_N1'] =  finalGraph.loc[idxLastFrame, 'objectID_N1'] - diff[i]

    return finalGraph

def getMinimumSource(graph,minAmount):
    graph = graph.reset_index(drop=True)
    graph = graph.sort_values(by=['objectID_N'])
    graph = graph.reset_index(drop=True)

    uniqueObjects = graph['objectID_N'].drop_duplicates().index.to_numpy()

    newGraph=pd.DataFrame(columns=['framesN','objectID_N','X_N','Y_N','Intensity_N','framesN1','objectID_N1','X_N1','Y_N1','Intensity_N1','edge_distance','intensity_difference','Diffusion Coefficient','Receptor Density','AP2',
'DR2','AP3','DR3','AP4','DR4','Labeled Fraction'])

    for i in range(len(uniqueObjects)):
        if i!= len(uniqueObjects)-1:
            graphEdges = graph.iloc[uniqueObjects[i]:uniqueObjects[i+1]].reset_index(drop=True)

            if len(graphEdges) > minAmount:
                smallestValues = graphEdges['edge_distance'].nsmallest(minAmount).index
                graphEdges = graphEdges.iloc[smallestValues]
        else:
            graphEdges = graph.iloc[uniqueObjects[i]:].reset_index(drop=True)
            if len(graphEdges) > minAmount:
                smallestValues = graphEdges['edge_distance'].nsmallest(minAmount).index
                graphEdges = graphEdges.iloc[smallestValues]
        newGraph = pd.concat([newGraph,graphEdges])
    newGraph = newGraph.reset_index(drop=True)

    return newGraph

def getMinimumTarget(graph,minAmount):
    graph = graph.reset_index(drop=True)
    graph = graph.sort_values(by=['objectID_N1'])
    graph = graph.reset_index(drop=True)

    uniqueObjects = graph['objectID_N1'].drop_duplicates().index.to_numpy()

    newGraph=pd.DataFrame(columns=['framesN','objectID_N','X_N','Y_N','Intensity_N','framesN1','objectID_N1','X_N1','Y_N1','Intensity_N1','edge_distance','intensity_difference','Diffusion Coefficient','Receptor Density','AP2',
'DR2','AP3','DR3','AP4','DR4','Labeled Fraction'])

    for i in range(len(uniqueObjects)):
        if i!= len(uniqueObjects)-1:
            graphEdges = graph.iloc[uniqueObjects[i]:uniqueObjects[i+1]].reset_index(drop=True)

            if len(graphEdges) > minAmount:
                smallestValues = graphEdges['edge_distance'].nsmallest(minAmount).index
                graphEdges = graphEdges.iloc[smallestValues]
        else:
            graphEdges = graph.iloc[uniqueObjects[i]:].reset_index(drop=True)
            if len(graphEdges) > minAmount:
                smallestValues = graphEdges['edge_distance'].nsmallest(minAmount).index
                graphEdges = graphEdges.iloc[smallestValues]
        newGraph = pd.concat([newGraph,graphEdges])
    newGraph = newGraph.reset_index(drop=True)

    return newGraph

def trimGraph(graphSource, graphTarget):
    combinedGraph = pd.concat([graphSource,graphTarget])
    combinedGraph=combinedGraph.drop_duplicates().reset_index(drop=True)
    combinedGraph=combinedGraph.sort_values(by=['objectID_N']).reset_index(drop=True)

    return combinedGraph

def GraphGeneratorTetramers(simulationData, r):

    """
    This function generates a graph with the information with each node/detections per frame

    Parameters
    ----------
    simulationData : simulationData obtained from
        receptorInfoLabeled.receptorTraj from receptorAggregationSimpleOptimized
        function in MATLAB
            pd.DataFrame (# of molecules per frame * # of frames, # of features)
    r : radius threshold (in same units as coordinates)
        int
    boundarySize : edge length of a square image (in same units as coordinates)
        int

    Returns
    -------
    finalGraph : table that shows all the edge connections and their respective information
    """
    simulationData = simulationData.dropna().reset_index(drop=True)
    objectID = range(len(simulationData))
    simulationData['ObjectID'] = objectID
    r = r
    index_frame = simulationData['Frame'].drop_duplicates().index.to_numpy()
    frames=[]

    for i in range(len(index_frame)):
        if i!=len(index_frame)-1:
            frames.append(simulationData.iloc[index_frame[i]:index_frame[i+1]])
        else:
            frames.append(simulationData.iloc[index_frame[i]:])
    graph=pd.DataFrame(columns=['framesN','objectID_N','X_N','Y_N','Intensity_N','framesN1','objectID_N1','X_N1','Y_N1','Intensity_N1','edge_distance','intensity_difference','Diffusion Coefficient','Receptor Density','AP2',
'DR2','AP3','DR3','AP4','DR4','Labeled Fraction'])

    for i in range(len(frames)-1):

        t_n = frames[i]
        t_n1 = frames[i+1]

        X_N_coord = t_n['X'].to_numpy()
        Y_N_coord = t_n['Y'].to_numpy()

        X_N1_coord = t_n1['X'].to_numpy()
        Y_N1_coord = t_n1['Y'].to_numpy()

        intensityN = t_n['Intensity'].to_numpy()
        intensityN1 = t_n1['Intensity'].to_numpy()

        intensityDifference = np.subtract.outer(intensityN1, intensityN).T

        objectID_N = t_n['ObjectID'].to_numpy()

        objectID_N1 = t_n1['ObjectID'].to_numpy()

        DC = t_n['Diffusion Coefficient'].to_numpy()
        RD = t_n['Receptor Density'].to_numpy()
        AP2 = t_n['AP2'].to_numpy()
        DR2 = t_n['DR2'].to_numpy()
        AP3 = t_n['AP3'].to_numpy()
        DR3 = t_n['DR3'].to_numpy()
        AP4 = t_n['AP4'].to_numpy()
        DR4 = t_n['DR4'].to_numpy()
        LF = t_n['Labeled Fraction'].to_numpy()

        edge_distances = cdist(np.array([X_N_coord, Y_N_coord]).T, np.array([X_N1_coord, Y_N1_coord]).T,'euclidean')
        find_edge_distance_index = np.where(edge_distances < r)

        X_N_coordFiltered = pd.DataFrame(X_N_coord[find_edge_distance_index[0]],columns=['X_N'])
        X_N1_coordFiltered = pd.DataFrame(X_N1_coord[find_edge_distance_index[1]], columns=['X_N1'])
        Y_N_coordFiltered = pd.DataFrame(Y_N_coord[find_edge_distance_index[0]],columns=['Y_N'])
        Y_N1_coordFiltered = pd.DataFrame(Y_N1_coord[find_edge_distance_index[1]], columns=['Y_N1'])
        edge_distanceFiltered = pd.DataFrame(edge_distances[find_edge_distance_index],columns=['edge_distance'])
        intensityNFiltered = pd.DataFrame(intensityN[find_edge_distance_index[0]], columns = ['Intensity_N'])
        intensityN1Filtered = pd.DataFrame(intensityN1[find_edge_distance_index[1]], columns = ['Intensity_N1'])
        framesN = pd.DataFrame((np.zeros(shape = (X_N_coordFiltered.shape[0],1)) + i).astype(int), columns = ['framesN'])
        framesN1 = pd.DataFrame((np.zeros(shape = (X_N_coordFiltered.shape[0],1)) + i + 1).astype(int), columns=['framesN1'])
        DCFiltered = pd.DataFrame(DC[find_edge_distance_index[0]],columns=['Diffusion Coefficient'])
        RDFiltered = pd.DataFrame(RD[find_edge_distance_index[0]],columns=['Receptor Density'])
        AP2Filtered = pd.DataFrame(AP2[find_edge_distance_index[0]],columns=['AP2'])
        DR2Filtered = pd.DataFrame(DR2[find_edge_distance_index[0]],columns=['DR2'])
        AP3Filtered = pd.DataFrame(AP3[find_edge_distance_index[0]],columns=['AP3'])
        DR3Filtered = pd.DataFrame(DR3[find_edge_distance_index[0]],columns=['DR3'])
        AP4Filtered = pd.DataFrame(AP4[find_edge_distance_index[0]],columns=['AP4'])
        DR4Filtered = pd.DataFrame(DR4[find_edge_distance_index[0]],columns=['DR4'])
        LFFiltered = pd.DataFrame(LF[find_edge_distance_index[0]],columns=['Labeled Fraction'])
        objectID_NFiltered = pd.DataFrame(objectID_N[find_edge_distance_index[0]],columns=['objectID_N'])
        objectID_N1Filtered = pd.DataFrame(objectID_N1[find_edge_distance_index[1]],columns=['objectID_N1'])
        intensityDifferenceFiltered = pd.DataFrame(intensityDifference[find_edge_distance_index],columns=['intensity_difference'])

        edges=pd.concat([framesN,objectID_NFiltered,X_N_coordFiltered,Y_N_coordFiltered,intensityNFiltered,framesN1,objectID_N1Filtered,X_N1_coordFiltered,Y_N1_coordFiltered,intensityN1Filtered,edge_distanceFiltered, intensityDifferenceFiltered,DCFiltered, RDFiltered, AP2Filtered, DR2Filtered,AP3Filtered, DR3Filtered,AP4Filtered, DR4Filtered,LFFiltered],axis=1)

        graph=pd.concat([graph,edges])

    graphSource = getMinimumSource(graph,3)
    graphTarget = getMinimumTarget(graph,3)
    trimmedGraph = trimGraph(graphSource,graphTarget)

    return trimmedGraph
