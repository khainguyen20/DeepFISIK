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
from deepfisik.models.gnn_oligo import *
import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.nn import CrossEntropyLoss
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd
import sys
from torch_geometric.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric.transforms as T

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred))
    return acc

def train(device,train_loader, model, time_step, optimizer, criterion,noClasses):
    model.train()

    losses = []

    i = 0
    for train_data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        tModel = time.time()
        data = train_data
        x=data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index.long()
        batch_index = data.batch
        if torch.cuda.device_count() > 1:
            node_out, edge_out, global_out, DC, AP, DR,RD,LF = model(x, edge_attr, edge_index, batch_index, device)

            true_DC = torch.cat([datalist.DC.reshape(1) for datalist in data]).to(DC.device)
            true_DC = torch.sqrt(2*true_DC*time_step)
            true_DC = true_DC.reshape(true_DC.size()[0],1)

            true_AP = torch.cat([datalist.AP.reshape(1) for datalist in data]).to(AP.device)
            true_AP = true_AP.reshape(true_AP.size()[0],1)

            true_DR = torch.cat([datalist.DR.reshape(1) for datalist in data]).to(DR.device)
            true_DR = true_DR.reshape(true_DR.size()[0],1)
        else:

            node_out, edge_out, u= model(data.x.to(device),data.edge_index.long().to(device),data.edge_attr.to(device), data.batch.to(device), device)

            if noClasses == 2:
                trueOligoLabel = data.oligoLabel.float().to(device)
                trueOligoLabel = trueOligoLabel.reshape(trueOligoLabel.size()[0],1)
            elif noClasses > 2:
                trueOligoLabel = data.oligoLabel.float().to(device)

        loss = criterion(u,trueOligoLabel)

        tBack = time.time()
        loss.backward()  # Derive gradients.
        tOpt = time.time()
        optimizer.step()  # Update parameters based on gradients.

        losses.append(loss.item())

    return np.mean(losses)

def test(device,loader, model, time_step,noClasses):

    model.eval()

    accuracy = []

    with torch.no_grad():
        for test_data in loader:
            data = test_data
            x=data.x
            edge_attr = data.edge_attr
            edge_index = data.edge_index.long()
            batch_index = data.batch

            if torch.cuda.device_count() > 1:
                node_out, edge_out, global_out, DC, AP, DR,RD,LF = model(x, edge_attr, edge_index, batch_index, device)

                true_DC = torch.cat([datalist.DC.reshape(1) for datalist in data]).to(DC.device)
                true_DC = torch.sqrt(2*true_DC*time_step)
                true_DC = true_DC.reshape(true_DC.size()[0],1)
                true_AP = torch.cat([datalist.AP.reshape(1) for datalist in data]).to(AP.device)
                true_AP = true_AP.reshape(true_AP.size()[0],1)
                true_DR = torch.cat([datalist.DR.reshape(1) for datalist in data]).to(DR.device)
                true_DR = true_DR.reshape(true_DR.size()[0],1)
            else:

                node_out, edge_out, u= model(data.x.to(device),data.edge_index.long().to(device),data.edge_attr.to(device), data.batch.to(device), device)

                if noClasses == 2:
                    pred = torch.round(torch.sigmoid(u))
                    trueOligoLabel = data.oligoLabel.float().to(device)
                    trueOligoLabel = trueOligoLabel.reshape(trueOligoLabel.size()[0],1)
                elif noClasses > 2:
                    pred = torch.argmax(torch.softmax(u,dim=1),dim=1)
                    trueOligoLabel = data.oligoLabel.float().to(device)
                    trueOligoLabel = torch.argmax(trueOligoLabel,dim=1)

            acc = accuracy_fn(trueOligoLabel,pred)

            accuracy.append(acc)

    return np.mean(accuracy)#, np.asarray(residuals_testDC).flatten(), np.asarray(residuals_testAP).flatten(), np.asarray(residuals_testDR).flatten()

def train_testOligo(
    device, train_loader, test_loader, model, time_step, model_params,savePath,date,checkpointFlag=False, edge_col_name=None,
    learning_rate=0.01, e_patience = 10, min_acc= 0.005, n_epochs=500, weight_decay = .01):
    t0 = time.time()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model(model_params = model_params), device_ids = [0,1,2,3])
        model = model.to(device)
        print("Model is using GPU parallelization")
    else:
        if checkpointFlag == False:
            model = model(model_params = model_params).to(device)
            print('Not using old model')
        elif checkpointFlag == True:
            print('Using previous model')
            model = model(model_params = model_params).to(device)
            pathCheckpoint = '/project/biophysics/jaqaman_lab/interKinetics/knguyen/GNN/2024/2024_09_30_Run/model2/freezeModel/modelGNN_2024_09_20_dr0_l6_lr1e-4_99.pt'
            checkpoint = torch.load(pathCheckpoint,map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Loaded model checkpoint: ',pathCheckpoint)
            print('Before: ')
            for name, param in model.named_parameters():
                print(name, param.requires_grad)

            for param in model.parameters():
                param.requires_grad = False

            for param in model.DCOut.parameters():
                param.requires_grad = True

            for param in model.APOut.parameters():
                param.requires_grad = True

            for param in model.DROut.parameters():
                param.requires_grad = True
            print('After: ')
            for name, param in model.named_parameters():
                print(name, param.requires_grad)

        if device.type == 'cuda':
            print("Using GPU")

    print(model)
    noClasses = model_params["model_noClasses"][0]

    if noClasses == 2:
        criterion = BCEWithLogitsLoss().to(device)
    elif noClasses > 2:
        criterion = CrossEntropyLoss().to(device)

    learnable_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(learnable_parameters, lr=learning_rate, weight_decay = weight_decay, amsgrad = True)
    print('Using ADAM optimizer')

    stepSize = 15
    gammaValue = .4

    print('stepSize: ', stepSize)
    print('gammaValue: ', gammaValue)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = stepSize, gamma = gammaValue)

    loss, train_score, test_score, test_pred, test_true = [], [], [], [], []

    for epoch in range(n_epochs):
        trainResults = train(device, train_loader, model, time_step, optimizer, criterion,noClasses)
        loss += [trainResults]

        train_validation = test(device, train_loader, model,time_step,noClasses)
        train_score += [train_validation]

        test_validation = test(device,test_loader, model, time_step,noClasses)
        test_score += [test_validation]

        scheduler.step()

        if (epoch+1)%10==0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss[-1]:.4f}, Train: {train_score[-1]:.4f}, Test: {test_score[-1]:.4f}')

        results = pd.DataFrame({
            'loss': loss,
            'train_score': train_score,
            'test_score': test_score,
            'time':(time.time()-t0)/60
        })

        results.to_csv(savePath+'/intermediateResults/IntermediateResults_'+date+'.csv')

        torch.save(model, savePath+'/models/modelGNN_'+date+'_'+str(epoch)+'.pt')

        if torch.cuda.device_count() > 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'loss_state_dict': criterion.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss[-1],
                }, savePath+'/checkpoints/modelGNN_'+date+'_'+str(epoch)+'.pt')
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss[-1],
                }, savePath+'/checkpoints/modelGNN_'+date+'_'+str(epoch)+'.pt')

    return model, results#, residualTrainCSV, trainAccuracyResidualCSV, testAccuracyResidualCSV
