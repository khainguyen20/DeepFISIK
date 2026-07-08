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
from deepfisik.models.gnn_tetramer import *
from deepfisik.losses.loss_tetramer import *
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
import pandas as pd
import sys
from torch_geometric.nn import DataParallel
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric.transforms as T

def train(device,train_loader, model, time_step, optimizer, criterion):
    model.train()

    lossesDC = []
    lossesAP2 = []
    lossesDR2 = []
    lossesAP3 = []
    lossesDR3 = []
    lossesAP4 = []
    lossesDR4 = []
    lossesRD = []
    lossesLF = []
    losses = []
    residuals_trainDC = []
    residuals_trainAP = []
    residuals_trainDR = []
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

            node_out, edge_out, global_out,DC,AP2,AP3,AP4,DR2,DR3,DR4,RD,LF = model(data.x.to(device),data.edge_index.long().to(device),data.edge_attr.to(device), data.batch.to(device), device)

            true_DC = data.DC.float().to(device)
            true_DC = torch.sqrt(2*true_DC*time_step)*30 #getting the true DC values
            true_DC = true_DC.reshape(true_DC.size()[0],1)
            true_AP2 = data.AP2.float().to(device)*50 # getting the true AP values
            true_AP2 = true_AP2.reshape(true_AP2.size()[0],1)
            true_DR2 = data.DR2.float().to(device)*20*time_step  # getting the true DR values
            true_DR2 = true_DR2.reshape(true_DR2.size()[0],1)
            true_AP3 = data.AP3.float().to(device)*50 # getting the true AP values
            true_AP3 = true_AP3.reshape(true_AP3.size()[0],1)
            true_DR3 = data.DR3.float().to(device)*20*time_step  # getting the true DR values
            true_DR3 = true_DR3.reshape(true_DR3.size()[0],1)
            true_AP4 = data.AP3.float().to(device)*50 # getting the true AP values
            true_AP4 = true_AP4.reshape(true_AP4.size()[0],1)
            true_DR4 = data.DR4.float().to(device)*20*time_step  # getting the true DR values
            true_DR4 = true_DR4.reshape(true_DR4.size()[0],1)
            true_RD = data.RD.float().to(device)
            true_RD = true_RD.reshape(true_RD.size()[0],1)
            true_LF = data.LF.float().to(device) * 10
            true_LF = true_LF.reshape(true_LF.size()[0],1)

        lossDC,lossAP2,lossAP3,lossAP4,lossDR2,lossDR3,lossDR4,lossRD,lossLF,loss = criterion([true_DC,true_AP2,true_AP3,true_AP4,true_DR2,true_DR3,true_DR4,true_RD,true_LF],[DC,AP2,AP3,AP4,DR2,DR3,DR4,RD,LF]) # calculate loss for DC, AP, DR, and summed losses

        tBack = time.time()
        loss.backward()  # Derive gradients.
        tOpt = time.time()
        optimizer.step()  # Update parameters based on gradients.

        losses.append(loss.item())
        lossesDC.append(lossDC.item())
        lossesAP2.append(lossAP2.item())
        lossesAP3.append(lossAP3.item())
        lossesAP4.append(lossAP4.item())
        lossesDR2.append(lossDR2.item())
        lossesDR3.append(lossDR3.item())
        lossesDR4.append(lossDR4.item())
        lossesRD.append(lossRD.item())
        lossesLF.append(lossLF.item())

    return np.mean(losses), np.mean(lossesDC), np.mean(lossesAP2),np.mean(lossesAP3),np.mean(lossesAP4), np.mean(lossesDR2),np.mean(lossesDR3),np.mean(lossesDR4),np.mean(lossesRD),np.mean(lossesLF)#, np.asarray(residuals_trainDC).flatten(), np.asarray(residuals_trainAP).flatten(), np.asarray(residuals_trainDR).flatten()

def test(device,loader, model, time_step,criterion):

    model.eval()

    lossesDC = []
    lossesAP2 = []
    lossesDR2 = []
    lossesAP3 = []
    lossesDR3 = []
    lossesAP4 = []
    lossesDR4 = []
    lossesRD = []
    lossesLF = []
    losses = []
    scores = []
    y_pred = 0
    y_true = 0
    results = []
    residuals_testDC = []
    residuals_testAP = []
    residuals_testDR = []

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

                node_out, edge_out, global_out,DC,AP2,AP3,AP4,DR2,DR3,DR4,RD,LF = model(data.x.to(device),data.edge_index.long().to(device),data.edge_attr.to(device), data.batch.to(device), device)

                true_DC = data.DC.float().to(device)
                true_DC = torch.sqrt(2*true_DC*time_step)*30 #getting the true DC values
                true_DC = true_DC.reshape(true_DC.size()[0],1)
                true_AP2 = data.AP2.float().to(device)*50 # getting the true AP values
                true_AP2 = true_AP2.reshape(true_AP2.size()[0],1)
                true_DR2 = data.DR2.float().to(device)*20*time_step  # getting the true DR values
                true_DR2 = true_DR2.reshape(true_DR2.size()[0],1)
                true_AP3 = data.AP3.float().to(device)*50 # getting the true AP values
                true_AP3 = true_AP3.reshape(true_AP3.size()[0],1)
                true_DR3 = data.DR3.float().to(device)*20*time_step  # getting the true DR values
                true_DR3 = true_DR3.reshape(true_DR3.size()[0],1)
                true_AP4 = data.AP4.float().to(device)*50 # getting the true AP values
                true_AP4 = true_AP4.reshape(true_AP4.size()[0],1)
                true_DR4 = data.DR4.float().to(device)*20*time_step  # getting the true DR values
                true_DR4 = true_DR4.reshape(true_DR4.size()[0],1)
                true_RD = data.RD.float().to(device)
                true_RD = true_RD.reshape(true_RD.size()[0],1)
                true_LF = data.LF.float().to(device) * 10
                true_LF = true_LF.reshape(true_LF.size()[0],1)

            lossDC,lossAP2,lossAP3,lossAP4,lossDR2,lossDR3,lossDR4,lossRD,lossLF,loss = criterion([true_DC,true_AP2,true_AP3,true_AP4,true_DR2,true_DR3,true_DR4,true_RD,true_LF],[DC,AP2,AP3,AP4,DR2,DR3,DR4,RD,LF]) # calculate loss for DC, AP, DR, and summed losses

            losses.append(loss.item())
            lossesDC.append(lossDC.item())
            lossesAP2.append(lossAP2.item())
            lossesAP3.append(lossAP3.item())
            lossesAP4.append(lossAP4.item())
            lossesDR2.append(lossDR2.item())
            lossesDR3.append(lossDR3.item())
            lossesDR4.append(lossDR4.item())
            lossesRD.append(lossRD.item())
            lossesLF.append(lossLF.item())

    accuracy = np.mean(losses)
    accuracyDC = np.mean(lossesDC)
    accuracyAP2 = np.mean(lossesAP2)
    accuracyDR2 = np.mean(lossesDR2)
    accuracyAP3 = np.mean(lossesAP3)
    accuracyDR3 = np.mean(lossesDR3)
    accuracyAP4 = np.mean(lossesAP4)
    accuracyDR4 = np.mean(lossesDR4)
    accuracyRD = np.mean(lossesRD)
    accuracyLF = np.mean(lossesLF)
    results.append(accuracy)
    results.append(accuracyDC)
    results.append(accuracyAP2)
    results.append(accuracyAP3)
    results.append(accuracyAP4)
    results.append(accuracyDR2)
    results.append(accuracyDR3)
    results.append(accuracyDR4)
    results.append(accuracyRD)
    results.append(accuracyLF)

    return results#, np.asarray(residuals_testDC).flatten(), np.asarray(residuals_testAP).flatten(), np.asarray(residuals_testDR).flatten()

def train_testTetramers(
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
    criterion = MultiMSETetramer().to(device)

    learnable_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(learnable_parameters, lr=learning_rate, weight_decay = weight_decay, amsgrad = True)
    print('Using ADAM optimizer')

    stepSize = 15
    gammaValue = .4

    print('stepSize: ', stepSize)
    print('gammaValue: ', gammaValue)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = stepSize, gamma = gammaValue)

    loss, train_score, test_score, test_pred, test_true = [], [], [], [], []

    lossDC = []
    lossAP2 = []
    lossDR2 = []
    lossAP3 = []
    lossDR3 = []
    lossAP4 = []
    lossDR4 = []
    lossRD = []
    lossLF = []

    train_scoreDC = []
    train_scoreAP2 = []
    train_scoreDR2 = []
    train_scoreAP3 = []
    train_scoreDR3 = []
    train_scoreAP4 = []
    train_scoreDR4 = []
    train_scoreRD = []
    train_scoreLF = []

    test_scoreDC = []
    test_scoreAP2 = []
    test_scoreDR2 = []
    test_scoreAP3 = []
    test_scoreDR3 = []
    test_scoreAP4 = []
    test_scoreDR4 = []
    test_scoreRD = []
    test_scoreLF = []

    residualTrainDC = []
    residualTrainAP = []
    residualTrainDR = []

    trainAccuracyResidualDC = []
    trainAccuracyResidualAP = []
    trainAccuracyResidualDR = []

    testAccuracyResidualDC = []
    testAccuracyResidualAP = []
    testAccuracyResidualDR = []

    for epoch in range(n_epochs):
        trainResults = train(device, train_loader, model, time_step, optimizer, criterion)
        loss += [trainResults[0]]
        lossDC += [trainResults[1]]
        lossAP2 += [trainResults[2]]
        lossAP3 += [trainResults[3]]
        lossAP4 += [trainResults[4]]
        lossDR2 += [trainResults[5]]
        lossDR3 += [trainResults[6]]
        lossDR4 += [trainResults[7]]
        lossRD += [trainResults[8]]
        lossLF += [trainResults[9]]
        train_validation = test(device, train_loader, model,time_step,criterion)
        trainAccuracy = train_validation#train_validation[0]
        train_score += [trainAccuracy[0]]
        train_scoreDC += [trainAccuracy[1]]
        train_scoreAP2 += [trainAccuracy[2]]
        train_scoreAP3 += [trainAccuracy[3]]
        train_scoreAP4 += [trainAccuracy[4]]
        train_scoreDR2 += [trainAccuracy[5]]
        train_scoreDR3 += [trainAccuracy[6]]
        train_scoreDR4 += [trainAccuracy[7]]
        train_scoreRD += [trainAccuracy[8]]
        train_scoreLF += [trainAccuracy[9]]
        test_validation = test(device,test_loader, model, time_step, criterion)
        testAccuracy = test_validation#test_validation[0]
        test_score += [testAccuracy[0]]
        test_scoreDC += [testAccuracy[1]]
        test_scoreAP2 += [testAccuracy[2]]
        test_scoreAP3 += [testAccuracy[3]]
        test_scoreAP4 += [testAccuracy[4]]
        test_scoreDR2 += [testAccuracy[5]]
        test_scoreDR3 += [testAccuracy[6]]
        test_scoreDR4 += [testAccuracy[7]]
        test_scoreRD += [testAccuracy[8]]
        test_scoreLF += [testAccuracy[9]]

        scheduler.step()

        if (epoch+1)%10==0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss[-1]:.4f}, Train: {train_score[-1]:.4f}, Test: {test_score[-1]:.4f}')

        results = pd.DataFrame({
            'loss': loss, 'lossDC':lossDC,'lossAP2':lossAP2,'lossAP3':lossAP3,'lossAP4':lossAP4,'lossDR2':lossDR2,'lossDR3':lossDR3,'lossDR4':lossDR4,'lossRD':lossRD,'lossLF':lossLF,
            'train_score': train_score, 'train_scoreDC': train_scoreDC,'train_scoreAP2': train_scoreAP2,'train_scoreAP3': train_scoreAP3,'train_scoreAP4': train_scoreAP4,'train_scoreDR2': train_scoreDR2,'train_scoreDR3': train_scoreDR3,'train_scoreDR4': train_scoreDR4,'train_scoreRD':train_scoreRD,'train_scoreLF':train_scoreLF,
            'test_score': test_score, 'test_scoreDC': test_scoreDC, 'test_scoreAP2': test_scoreAP2,'test_scoreAP3': test_scoreAP3,'test_scoreAP4': test_scoreAP4, 'test_scoreDR2': test_scoreDR2,'test_scoreDR3': test_scoreDR3,'test_scoreDR4': test_scoreDR4,'test_scoreRD':test_scoreRD,'test_scoreLF':test_scoreLF,
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

def visualize_loss(results):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=results.index, y=results['train_score'], name = 'train_score'))
    fig.add_trace(go.Scatter(x=results.index, y=results['test_score'], name = 'test_score'))
    fig.add_trace(go.Scatter(x=results.index, y=results['loss'], name = 'loss'))

    fig.update_yaxes(title_text='Score')
    fig.update_xaxes(title_text="Epoch")

    fig.update_yaxes(range=[0,1.3*max([max(results[c]) for c in ['loss', 'train_score', 'test_score']])])

    return fig
