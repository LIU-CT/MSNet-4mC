"""
@author: liu chunting
Department of IST, Kyoto University
"""

import argparse
import sys
import numpy as np
import torch
import torch.utils.data as Data

from model import modeltest as model
from sk_metrics import calculate_metrics_test
from loading_encoding import dataload, get_df_fastas, BINARY

def prediction(dataset = None, species = None, fasta_file = None):
    mydataset_list = ["Lin_2017_Dataset", "Li_2020_Dataset"]
    
    dic_specis = {"A.thaliana":"A",     
                "C.elegans":"C",
                "D.melanogaster":"D", 
                "E.coli":"E",
                "G.pickeringii":"GP",  
                "G.subterraneus":"GS"}
    if dataset in mydataset_list:
        if fasta_file == None:
            print("Test on the benchmark")
            txt_test = dataload(dataset, species, None)
            load_path = "{}/Models/fine_tuning/Model_{}.pth".format(dataset, dic_specis.get(species))
        else: 
            print("No need for input files")
            sys.exit(1)
    elif dataset == "User_Dataset":
        if fasta_file == None:
            print("Missing input files")
            sys.exit(1)
        else:
            print("Test the user's dataset")
            txt_test = dataload("User_Dataset", species, fasta_file)
            load_path = "Li_2020_Dataset/Models/fine_tuning/Model_{}.pth".format(dic_specis.get(species))
    
    df_fastas_test = get_df_fastas(txt_test)    
    
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
        
    print("-------Test on {} {}-------".format(dataset, species))

    test_data_size = len(test_dataset)
    print("lenth of test datasets --{}".format(test_data_size))
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modeltest = model(load_pretrain = True, load_path = load_path)
    #modeltest = modeltest.to(device)
    
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
            #batchtest_X = batchtest_X.to(device)
            #batchtest_y = batchtest_y.to(device)
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
            
    
