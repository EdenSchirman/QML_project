import glob
import os
from os import listdir
import numpy as np

# dist_name_vec = ['log-normal','triangular','bimodal']
dist_name_vec = ['log-normal']
K_vec =['1','2']
bool_vec = ['True', 'False']
entnagler = 'circular'
for _,distribution_name in enumerate(dist_name_vec):
    for _,K in enumerate(K_vec):
        for _,third_layer in enumerate(bool_vec):

            #directory_name = '{}/P={},K={},3rd_layer={},dropouts=False,n_hidden0=50,n_hidden1=20,N_epochs=700/'.format(distribution_name,distribution_name,K,third_layer)
            #directory_name = 'P={},K={},3rd_layer={},enatngler={},n_hidden0=50,n_hidden1=20,N_epochs=700/'.format(distribution_name,K,third_layer,entnagler)
            directory_name = 'P={},K={},3rd_layer={},enatngler={},n_hidden0=512,n_hidden1=256,N_epochs=700/'.format(distribution_name,K,third_layer,entnagler)
            
            perfix = '/Users/edenschirman/git/QML_project/Figures/'
            re_vec_list = glob.glob(perfix+directory_name+'RE/*.npy')

            last_re_val =[]
            min_re_val=[]
            for i,re_vec_str in enumerate(re_vec_list):
                re_vec = np.load(re_vec_str)
                last_re_val.append(re_vec[-1])
                min_re_val.append(np.min(re_vec))

            last_re_val = np.array(last_re_val)
            min_re_val = np.array(min_re_val)

            mu_last = np.mean(last_re_val)
            std_last = np.std(last_re_val)

            mu_min = np.mean(min_re_val)
            std_min = np.std(min_re_val)

            filename_last = 'RE[-1], mu={:.4f},std={:.4f}.npy'.format(mu_last,std_last)
            filename_min = 'min RE, mu={:.4f},std={:.4f}.npy'.format(mu_min,std_min)

            np.save(perfix + directory_name + filename_last, last_re_val)
            np.save(perfix + directory_name + filename_min, min_re_val)



