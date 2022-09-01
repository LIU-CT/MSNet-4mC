"""
@author: liu chunting
Department of IST, Kyoto University
"""
### calculate the class weights ###
import math
import numpy as np

train_A = 3692
train_C = 2900
train_D = 3302
train_E = 724
train_GP = 1062
train_GS = 1690


num_ls = [train_A , train_C , 
          train_D , train_E , 
          train_GP , train_GS ]

train_sum = train_A + train_C \
            + train_D + train_E \
            + train_GP + train_GS 

freq_all = []

for i in range(len(num_ls)):
    freq_i = num_ls[i] / train_sum 
    freq_all.append( freq_i )

weight_class_revised_2 = []

for i in range(len(num_ls)):
    weight_i = math.log(1 / freq_all[i])
    weight_class_revised_2.append( weight_i )
   
weight_class_revised_2 = np.array(weight_class_revised_2)#array([1.28684507, 1.52830265, 1.39848505, 2.91597728, 2.53285947, 2.06828486])
#np.save('class_weight.npy', weight_class_revised_2)   
