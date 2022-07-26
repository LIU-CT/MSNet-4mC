"""
Created on Sat May  7 00:36:22 2022

@author: liu chunting
Department of IST, Kyoto University
"""

import numpy as np
import torch
import torch.utils.data as Data
from sk_metrics import calculate_metrics_test

from model import modeltest as model
from data_load_encoding_DT2 import load_data_DT2, add_labels, get_df_fastas_DT2, BINARY, BINARY_w_labels_DT2

#from calculate_metrics_plot import calculate_metrics_FT as getFT
#from calculate_metrics_plot import plt_roc_all_in_one

########## load txt ##########
train_N_txt, train_P_txt, test_N_txt, test_P_txt = load_data_DT2()

########## txt of test ##########
txt_A_test = test_N_txt[0] + test_P_txt[0]
txt_C_test = test_N_txt[1] + test_P_txt[1]
txt_D_test = test_N_txt[2] + test_P_txt[2]
txt_E_test = test_N_txt[3] + test_P_txt[3]
txt_GP_test = test_N_txt[4] + test_P_txt[4]
txt_GS_test = test_N_txt[5] + test_P_txt[5]

########## load model ##########
'''
##########  base model ########## 
ls_txt_name =['basemodel','A','C','D','E','GP','GS']

txt_ALL_test = txt_A_test + txt_C_test +txt_D_test + txt_E_test + txt_GP_test + txt_GS_test
data_txt_test_ls = [txt_ALL_test,    
                    txt_A_test, 
                    txt_C_test, 
                    txt_D_test, 
                    txt_E_test, 
                    txt_GP_test, 
                    txt_GS_test]

load_path = 'Models/base_model/Model_basemodel.pth'
'''
########## fine-tuning model ########## 
ls_txt_name =['A','C','D','E','GP','GS']

data_txt_test_ls = [txt_A_test, 
                    txt_C_test, 
                    txt_D_test, 
                    txt_E_test, 
                    txt_GP_test, 
                    txt_GS_test]

load_path = ['Models/fine_tuning/Model_A.pth', 
             'Models/fine_tuning/Model_C.pth', 
             'Models/fine_tuning/Model_D.pth', 
             'Models/fine_tuning/Model_E.pth', 
             'Models/fine_tuning/Model_GP.pth', 
             'Models/fine_tuning/Model_GS.pth', ]


fpr_all = [(), (), (), (), (), ()]
tpr_all = [(), (), (), (), (), ()] 
AUC_all = [(), (), (), (), (), ()]

for num in range(len(data_txt_test_ls)):
    data_txt_test = data_txt_test_ls[num]
########## sequence & target ##########
    df_fastas_test = get_df_fastas_DT2(data_txt_test)    
    
    sequence_test = df_fastas_test['sequence']

    test_X_encodings = BINARY(sequence_test)
    test_X = np.array(test_X_encodings)
    test_X = test_X.transpose(0, 2, 1)
    
    target_test = df_fastas_test['target']
    
    test_y_target = []
    for i in target_test:
        test_y_target.append(i)
    test_y = np.array(test_y_target)
        
########## DataLoader ##########
########## test data preparing ##########
    test_X_tensor = torch.FloatTensor(test_X)
    test_y_tensor = torch.LongTensor(test_y)
    
    test_dataset = Data.TensorDataset(test_X_tensor, test_y_tensor)
    testloader = Data.DataLoader(test_dataset, batch_size=128, shuffle = False) # default: 128
        
    print("-------Test on {} -------".format(ls_txt_name[num]))

    test_data_size = len(test_dataset)
    print("lenth of test datasets：{}".format(test_data_size))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
########## path of base model ########## 
    #modeltest = model(load_pretrain = True, load_path = load_path)
########## path of scratch model or fine-tuning ########## 
    modeltest = model(load_pretrain = True, load_path = load_path[num])
    modeltest = modeltest.to(device)
    
    # test
    modeltest.eval()
    total_test_loss = 0
    total_accuracy = 0
    y_pred = []
    y_real = []
    outputs_all = []
    with torch.no_grad():
        for batch_test in testloader:
            batchtest_X, batchtest_y = batch_test
            batchtest_X = batchtest_X.to(device)
            batchtest_y = batchtest_y.to(device)
            outputs = modeltest(batchtest_X)
            
            # preserve
            outputs_all.append(outputs) 
                        
            temp1 = outputs.argmax(1).cpu().numpy().tolist()
            y_pred = y_pred + temp1
            
            temp2 = batchtest_y.cpu().numpy().tolist()
            y_real = y_real + temp2
            
            # calculate accuracy
            accuracy = (outputs.argmax(1) == batchtest_y).sum()
            total_accuracy = total_accuracy + accuracy
        
    calculate_metrics_test(y_real, y_pred, outputs_all)
    print("ACC on the test set: {}".format(total_accuracy/test_data_size))
    #fpr_all[num], tpr_all[num], AUC_all[num] = getFT(y_real, y_pred, outputs_all)

#plt_roc_all_in_one(fpr_all, tpr_all, AUC_all)



