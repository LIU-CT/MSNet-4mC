"""
@author: liu chunting
Department of IST, Kyoto University
"""

import numpy as np
import time

import torch
import torch.utils.data as Data
import torch.optim as optim

from torch import nn
from sk_metrics import calculate_metrics
from lr_scheduler import LR_Scheduler
from losses_for_training import CE_Loss

from model import modeltest as model
from data_load_encoding import load_data_DT1, add_labels, get_df_fastas_DT1, BINARY, BINARY_w_labels_DT1

# use cudnn
torch.backends.cudnn.benchmark = True  

########## load txt ##########
train_N_txt, train_P_txt, test_N_txt, test_P_txt = load_data_DT1()

########## add labels for training dataset ##########
train_N_txt_revised = add_labels(train_N_txt)
train_P_txt_revised = add_labels(train_P_txt)

########## txt of training set ##########
txt_A_train = train_N_txt_revised[0] + train_P_txt_revised[0]
txt_C_train = train_N_txt_revised[1] + train_P_txt_revised[1]
txt_D_train = train_N_txt_revised[2] + train_P_txt_revised[2]
txt_E_train = train_N_txt_revised[3] + train_P_txt_revised[3]
txt_GP_train = train_N_txt_revised[4] + train_P_txt_revised[4]
txt_GS_train = train_N_txt_revised[5] + train_P_txt_revised[5]

########## txt of test set ##########
txt_A_test = test_N_txt[0] + test_P_txt[0]
txt_C_test = test_N_txt[1] + test_P_txt[1]
txt_D_test = test_N_txt[2] + test_P_txt[2]
txt_E_test = test_N_txt[3] + test_P_txt[3]
txt_GP_test = test_N_txt[4] + test_P_txt[4]
txt_GS_test = test_N_txt[5] + test_P_txt[5]

ls_txt_name =['A','C','D','E','GP','GS']

data_txt_train_ls = [txt_A_train, 
                    txt_C_train, 
                    txt_D_train,
                    txt_E_train, 
                    txt_GP_train, 
                    txt_GS_train]

data_txt_test_ls = [txt_A_test, 
                    txt_C_test, 
                    txt_D_test, 
                    txt_E_test, 
                    txt_GP_test, 
                    txt_GS_test]
''' 
########## training from scrach ##########
load_pretrain = False
load_path = None
'''
########## fine-tuning on base model ##########
load_pretrain = True
load_path = 'Models/base_model/Model_basemodel.pth'    
    
temp_acc = []

for num in range(len(data_txt_test_ls)):
    ############## drop_s ######### A,C,D: 0.5 ######## E, GP, GS:0.8
    if num < 3:
        drop_s = 0.5
    else:
        drop_s = 0.8
    data_txt_train = data_txt_train_ls[num]
    data_txt_test = data_txt_test_ls[num]

########## sequence & target ##########
    df_fastas_train = get_df_fastas_DT1(data_txt_train)
    df_fastas_test = get_df_fastas_DT1(data_txt_test)

    sequence_train = df_fastas_train['sequence']
    sequence_test = df_fastas_test['sequence']

    #train_X_encodings = BINARY(sequence_train)
    train_X_encodings = BINARY_w_labels_DT1(sequence_train)
    test_X_encodings = BINARY(sequence_test)
    train_X = np.array(train_X_encodings)
    train_X = train_X.transpose(0, 2, 1)
    test_X = np.array(test_X_encodings)
    test_X = test_X.transpose(0, 2, 1)

    target_train = df_fastas_train['target']
    target_test = df_fastas_test['target']
    
    train_y_target = []
    test_y_target = []
    for i in target_train:
        train_y_target.append(i)
    
    for i in target_test:
        test_y_target.append(i)
    
    train_y = np.array(train_y_target)
    test_y = np.array(test_y_target)
    
########## DataLoader ##########
########## training data preparing ##########
    train_X_tensor = torch.FloatTensor(train_X)
    train_y_tensor = torch.LongTensor(train_y)
    
    train_dataset = Data.TensorDataset(train_X_tensor, train_y_tensor)
    trainloader = Data.DataLoader(train_dataset, batch_size=256, shuffle = True, num_workers = 0) # default: 256
    
########## test data preparing ##########
    test_X_tensor = torch.FloatTensor(test_X)
    test_y_tensor = torch.LongTensor(test_y)
    
    test_dataset = Data.TensorDataset(test_X_tensor, test_y_tensor)
    testloader = Data.DataLoader(test_dataset, batch_size=128, shuffle = True, num_workers = 0) # default: 128
    
    print("-------Train on {} -------".format(ls_txt_name[num]))
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print("lenth of train datasets：{}".format(train_data_size))
    print("lenth of test datasets：{}".format(test_data_size))    
    
    best_acc = 0
    save_path = '00_results/'# training from scratch
    #save_path = '00_results/fine_tuning/'# fine-tuning
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modeltest = model(load_pretrain = load_pretrain, load_path = load_path, drop_s = drop_s)
    ####base model:load_pretrain = False
    ####fine-tuning:load_pretrain = True
    modeltest = modeltest.to(device) 
    # Loss
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    # optimizer
    learning_rate = 1e-02 
    optimizer = optim.SGD(modeltest.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-04)
    
    # Epochs
    total_train_step = 0
    total_test_step = 0

    warm = 20 
    epochs = 300     
    
    scheduler = LR_Scheduler('cos', learning_rate, epochs, len(trainloader), warmup_epochs=warm)

    start_time = time.time()
    for epoch in range(epochs):
        print("-------Round {} -------".format(epoch + 1))
        y_pred_train = []
        y_real_train = []
        total_accuracy_train = 0
    
        # train
        modeltest.train()
        
        counter = 0 ### cos ###    
        
        for batch_train in trainloader: 
            batch_X_box, batch_y = batch_train
            batch_X, sign_X = batch_X_box[:,:,:41], batch_X_box[:,:,-1]
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            sign_X = sign_X.to(device)
            outputs = modeltest(batch_X)
            loss = loss_fn(outputs, batch_y)           
            
            scheduler(optimizer, counter, epoch) ### cos ###
            #print('lr:', optimizer.param_groups[0]['lr'])
            counter = counter + 1
    
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("number of training：{}, Loss: {}".format(total_train_step, loss.item()))

            temp1 = outputs.argmax(1).cpu().numpy().tolist()
            y_pred_train = y_pred_train + temp1
            
            temp2 = batch_y.cpu().numpy().tolist()
            y_real_train = y_real_train + temp2
            
            # calculate accuracy
            accuracy_train = (outputs.argmax(1) == batch_y).sum()
            total_accuracy_train = total_accuracy_train + accuracy_train
           
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
                
        # Save checkpoint.
        acc = (total_accuracy/test_data_size).detach().cpu().numpy()
        if acc > best_acc:
            print('Saving..')
            best_acc = acc
            print("-------Test on {} -------".format(ls_txt_name[num]))
            np.save(save_path + 'best_result{}.npy'.format(ls_txt_name[num]), best_acc)
            
            torch.save(modeltest.state_dict(), save_path + 'Model_{}.pth'.format(ls_txt_name[num]))    
    
        calculate_metrics(y_real, y_pred, outputs_all)
        print("ACC on the test set: {}".format(total_accuracy/test_data_size))
        print("Test on {} ".format(ls_txt_name[num])+'best acc:', best_acc)
        print("ACC on the train set: {}".format(total_accuracy_train/train_data_size))
    
    temp_acc.append('{}'.format(ls_txt_name[num])+': '+ str(best_acc)+'\n')
        
    with open(save_path +'best_acc.txt','w') as f:
        strp = ''
        f.write(strp.join(temp_acc))

