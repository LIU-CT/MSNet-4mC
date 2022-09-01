"""
@author: liu chunting
Department of IST, Kyoto University
"""
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import math
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import torch.nn.functional as F
import torch

def calculate_metrics(y_real, y_pred,outputs_all):

    confus_matrix = confusion_matrix(y_real, y_pred)
    
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN + 1e-20) 
    SP = TN / ( TN + FP + 1e-20)
    PR = TP / ( TP + FP + 1e-20)

    ACC = ( TP + TN ) / ( TP + TN + FN + FP + 1e-20)
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )) + 1e-20)
    F1_score = (2 * SN * PR)/(SN + PR + 1e-20)
    print("Model score --- SN:{0:<20}, SP:{1:<20}, PR:{2:<20}, ".format(SN, SP, PR))
    print("Model score --- ACC:{0:<20}, MCC:{1:<20}, F1:{2:<20} ".format(ACC, MCC,F1_score))
    
    # AUC
    auc_map = outputs_all[0]
    for i in range(1, len(outputs_all)):
        auc_map = torch.cat([auc_map, outputs_all[i]], dim=0)
        
    y_pred_prob = F.softmax(auc_map, dim=1).cpu().numpy()
    fpr, tpr, thres = roc_curve(y_real, y_pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)
    print('Model score --- AUC:', AUC)
    
def calculate_metrics_test(y_real, y_pred,outputs_all):
    confus_matrix = confusion_matrix(y_real, y_pred)
    
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN + 1e-20) 
    SP = TN / ( TN + FP + 1e-20)
    PR = TP / ( TP + FP + 1e-20)

    ACC = ( TP + TN ) / ( TP + TN + FN + FP + 1e-20)
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )) + 1e-20)
    F1_score = (2 * SN * PR)/(SN + PR + 1e-20)

    # AUC
    auc_map = outputs_all[0]
    for i in range(1, len(outputs_all)):
        auc_map = torch.cat([auc_map, outputs_all[i]], dim=0)
        
    y_pred_prob = F.softmax(auc_map, dim=1).cpu().numpy()
    fpr, tpr, thres = roc_curve(y_real, y_pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)

    print("Model score --- AUC:  , MCC:  , ACC:  , F1:   , SN:   , SP:  ,PR:  ")
    print("Model score --- {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f} ".format(AUC, MCC, ACC, F1_score, SN, SP, PR))

def calculate_metrics_test_1(y_real, y_pred,outputs_all):
    confus_matrix = confusion_matrix(y_real, y_pred)
    
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN + 1e-20) 
    SP = TN / ( TN + FP + 1e-20)
    PR = TP / ( TP + FP + 1e-20)

    ACC = ( TP + TN ) / ( TP + TN + FN + FP + 1e-20)
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )) + 1e-20)
    F1_score = (2 * SN * PR)/(SN + PR + 1e-20)

    # AUC
    auc_map = outputs_all[0]
    for i in range(1, len(outputs_all)):
        auc_map = torch.cat([auc_map, outputs_all[i]], dim=0)
        
    y_pred_prob = F.softmax(auc_map, dim=1).cpu().numpy()
    fpr, tpr, thres = roc_curve(y_real, y_pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)
    print("Model score --- MCC:  , ACC:  , SN:   , SP:  , PR:  , AUC:  ")
    print("Model score --- {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(MCC, ACC,  SN, SP, PR, AUC))  
    print("Model score --- F1_score:{0:.4f}".format(F1_score))  

def calculate_metrics_test_for_excel(y_real, y_pred,outputs_all):
    confus_matrix = confusion_matrix(y_real, y_pred)
    
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN + 1e-20) 
    SP = TN / ( TN + FP + 1e-20)
    PR = TP / ( TP + FP + 1e-20)

    ACC = ( TP + TN ) / ( TP + TN + FN + FP + 1e-20)
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )) + 1e-20)
    F1_score = (2 * SN * PR)/(SN + PR + 1e-20)

    # AUC
    auc_map = outputs_all[0]
    for i in range(1, len(outputs_all)):
        auc_map = torch.cat([auc_map, outputs_all[i]], dim=0)
        
    y_pred_prob = F.softmax(auc_map, dim=1).cpu().numpy()
    fpr, tpr, thres = roc_curve(y_real, y_pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)

    print("Model score --- MCC:  , ACC:  ,F1:   , SN:   , SP:  , PR:  , AUC:  ")
    print("Model score --- {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}".format(MCC, ACC, F1_score, SN, SP, PR, AUC))  
        
    return MCC, ACC, F1_score, SN, SP, PR, AUC
'''
def calculate_metrics_test_cross_species(y_real, y_pred,outputs_all):
    confus_matrix = confusion_matrix(y_real, y_pred)
    
    TN, FP, FN, TP = confus_matrix.ravel()
    
    SN = TP / ( TP + FN + 1e-20) 
    SP = TN / ( TN + FP + 1e-20)
    PR = TP / ( TP + FP + 1e-20)

    ACC = ( TP + TN ) / ( TP + TN + FN + FP + 1e-20)
    MCC = (( TP * TN ) - ( FP * FN )) / (math.sqrt(( TP + FN ) * ( TP + FP ) * ( TN + FP ) * ( TN + FN )) + 1e-20)
    F1_score = (2 * SN * PR)/(SN + PR + 1e-20)

    # AUC
    auc_map = outputs_all[0]
    for i in range(1, len(outputs_all)):
        auc_map = torch.cat([auc_map, outputs_all[i]], dim=0)
        
    y_pred_prob = F.softmax(auc_map, dim=1).cpu().numpy()
    fpr, tpr, thres = roc_curve(y_real, y_pred_prob[:, 1], pos_label=1)
    AUC = auc(fpr, tpr)

    print("Model score --- MCC:  , ACC:  , SN:   , SP:  , PR:  , AUC:  ")
    print("Model score --- {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}".format(MCC, ACC,  SN, SP, PR, AUC))    
    return ACC
'''          
