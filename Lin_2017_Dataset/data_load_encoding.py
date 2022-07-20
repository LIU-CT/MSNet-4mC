"""
@author: liu chunting
Department of IST, Kyoto University
"""
import pandas as pd
import numpy as np

########## load txt ##########
def load_data_DT1():    
########## negative txt for training ##########
    ls_train_N_txt = ['Training-datasets/NA_train.txt',  
                    'Training-datasets/NC_train.txt', 
                    'Training-datasets/ND_train.txt',
                    'Training-datasets/NE_train.txt',
                    'Training-datasets/NGpick_train.txt',
                    'Training-datasets/NGsub_train.txt'
                    ]
    
    train_N_txt = []
    for i in range(len(ls_train_N_txt)):
        train_N_txt.append([])
    
    for i in range(len(ls_train_N_txt)):
        with open(ls_train_N_txt[i],"r") as f:    
            train_N_txt[i] = f.readlines()    
    
    for i in range(len(train_N_txt)):
        print(len(train_N_txt[i]))        
    print('\n')
    
########## positive txt for training ##########
    ls_train_P_txt = ['Training-datasets/PA_train.txt',  
                    'Training-datasets/PC_train.txt', 
                    'Training-datasets/PD_train.txt',
                    'Training-datasets/PE_train.txt',
                    'Training-datasets/PGpick_train.txt',
                    'Training-datasets/PGsub_train.txt'
                    ]
    
    train_P_txt = []
    for i in range(len(ls_train_P_txt)):
        train_P_txt.append([])
    
    for i in range(len(ls_train_P_txt)):
        with open(ls_train_P_txt[i],"r") as f:    
            train_P_txt[i] = f.readlines()     
    
    for i in range(len(train_P_txt)):
        print(len(train_P_txt[i]))
    print('\n')
    
########## negative txt for testing ###########
    ls_test_N_txt = ['Testing-datasets/NA_test.txt',  
                    'Testing-datasets/NC_test.txt', 
                    'Testing-datasets/ND_test.txt',
                    'Testing-datasets/NE_test.txt',
                    'Testing-datasets/NGpick_test.txt',
                    'Testing-datasets/NGsub_test.txt'
                    ]
    
    test_N_txt = []
    for i in range(len(ls_test_N_txt)):
        test_N_txt.append([])
    
    for i in range(len(ls_test_N_txt)):
        with open(ls_test_N_txt[i],"r") as f:    
            test_N_txt[i] = f.readlines()     
    
    for i in range(len(test_N_txt)):
        print(len(test_N_txt[i]))
    print('\n')
        
########## positive txt for testing ##########
    ls_test_P_txt = ['Testing-datasets/PA_test.txt',  
                    'Testing-datasets/PC_test.txt', 
                    'Testing-datasets/PD_test.txt',
                    'Testing-datasets/PE_test.txt',
                    'Testing-datasets/PGpick_test.txt',
                    'Testing-datasets/PGsub_test.txt'
                    ]
    
    test_P_txt = []
    for i in range(len(ls_test_P_txt)):
        test_P_txt.append([])
    
    for i in range(len(ls_test_P_txt)):
        with open(ls_test_P_txt[i],"r") as f:    
            test_P_txt[i] = f.readlines()    
    
    for i in range(len(test_P_txt)):
        print(len(test_P_txt[i]))
    print('\n')
    
    return train_N_txt, train_P_txt, test_N_txt, test_P_txt


########## sequence & label ##########
def get_df_fastas_DT1(data_txt):
    fastas = []
    
    for i in range(0, len(data_txt), 2):    
        odd_row = data_txt[i]
        even_row = data_txt[i+1]
                        
        seq = even_row.split('\n')[0]
        if odd_row[1] == 'P':
            target = 1   
        elif odd_row[1] == 'N':
            target = 0  
        else: 
            print("error in datasets")        
        fastas.append([seq, target]) 
    
    df_fastas = pd.DataFrame(fastas)
    df_fastas.columns = ['sequence', 'target']
    return df_fastas

########## add class labels for training set ##########
def add_labels(train_txt):
    train_txt_revised = train_txt
    ls_label = [',a',',b',',c',',d',',e',',f']    
    
    for num_spe in range(len(train_txt)):
        for i in range(len(train_txt[num_spe])):
            if np.mod(i,2) == 1:
                train_txt_revised[num_spe][i] = train_txt[num_spe][i].split('\n')[0] + ls_label[num_spe] + '\n'
    
    return train_txt_revised

########## one-hot encoding ##########
def BINARY(sequence):
    encodings = []
    for seq in sequence:
        code = []
        for aa in seq:
            if aa == '-':
                code.append([0, 0, 0, 0])
            if aa == 'A':
                code.append([1, 0, 0, 0])
            if aa == 'C':
                code.append([0, 1, 0, 0])
            if aa == 'G':
                code.append([0, 0, 1, 0])
            if aa == 'T':
                code.append([0, 0, 0, 1])
        encodings.append(code)
    return encodings

########## one-hot encoding & class label ##########
def BINARY_w_labels_DT1(sequence):
    encodings = []
    ls_label = ['a','b','c','d','e','f']
    for seq in sequence:
        code = []
        for aa in seq:
            if aa == '-':
                code.append([0, 0, 0, 0])
            if aa == 'A':
                code.append([1, 0, 0, 0])
            if aa == 'C':
                code.append([0, 1, 0, 0])
            if aa == 'G':
                code.append([0, 0, 1, 0])
            if aa == 'T':
                code.append([0, 0, 0, 1])
            if aa in ls_label:
                for i in range(len(ls_label)):
                    if aa == ls_label[i]:
                        code.append([i,i,i,i])

        encodings.append(code)
    return encodings

