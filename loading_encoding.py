"""
@author: liu chunting
Department of IST, Kyoto University
"""
import re, os, sys
import pandas as pd
import numpy as np

def dataload(dataset = None, species = None, fasta_file = None):
    
    DT1_name = {"A.thaliana":"A_test",     
                "C.elegans":"C_test",
                "D.melanogaster":"D_test", 
                "E.coli":"E_test",
                "G.pickeringii":"Gpick_test",  
                "G.subterraneus":"Gsub_test"}
    
    DT2_name = {"A.thaliana":"Arabidopsis_thaliana",     
                "C.elegans":"Caenorhabditis_elegans",
                "D.melanogaster":"Drosophila_melanogaster", 
                "E.coli":"Escherichia_coli",
                "G.pickeringii":"Geobacter_pickeringii",  
                "G.subterraneus":"Geoalkalibacter_subterraneus"}
    if fasta_file == None:
        if dataset == "Lin_2017_Dataset":
            ls_test_N_txt = "Lin_2017_Dataset/Testing-datasets/N{}.txt".format(DT1_name.get(species))
            ls_test_P_txt = "Lin_2017_Dataset/Testing-datasets/P{}.txt".format(DT1_name.get(species))
            print(ls_test_N_txt)
            print(ls_test_P_txt)
            
        elif dataset == "Li_2020_Dataset":
            ls_test_N_txt = "Li_2020_Dataset/Testing-datasets/{}-test-N.txt".format(DT2_name.get(species))
            ls_test_P_txt = "Li_2020_Dataset/Testing-datasets/{}-test-P.txt".format(DT2_name.get(species))
            print(ls_test_N_txt)
            print(ls_test_P_txt)
            
        with open(ls_test_N_txt,"r") as f:    
            test_N_txt = f.readlines()    
        print(len(test_N_txt)/2)
        with open(ls_test_P_txt,"r") as f:    
            test_P_txt = f.readlines()    
        print(len(test_P_txt)/2)  
        txt_test = test_N_txt + test_P_txt 
        
    elif fasta_file != None:
        ## check user's fasta file
        if not os.path.exists(fasta_file):
            print('Error: "' + fasta_file + '" does not exist.')
            sys.exit(1)
    
        with open(fasta_file) as f:
            txt_test = f.readlines()
        
        for i_line in range(0, len(txt_test), 2):
            if txt_test[i_line][0] != ">":
            #if re.search('>', txt_test) is None:
                print('The input file seems not in fasta format.')
                sys.exit(1)
        #print(txt_test)
        print(len(txt_test)/2)
    return txt_test
        

    
def get_df_fastas(data_txt):
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


