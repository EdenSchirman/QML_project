from main import main
import numpy as np


# running one configuration of main
def run_one_config_of_main(network_params):
   # parameters
   n2avg = 5  
   num_not_converged = 0
   min_re_vec = []
   seed = [767, 3, 13, 94, 123]

   for i in range(n2avg):
      network_params['seed']=seed[i]
      re_vec, directory_name = main(network_params)  
      if re_vec[0] + 0.03 < re_vec[-1]:
         num_not_converged+=1
         continue
      # network_params['params_g'] = params_g
      min_re_vec.append(np.min(re_vec))

      print('\n ***************************** \n Finished run {}/{} \n rel_entropy = {} \n **************************** \n'.format(i+1, n2avg, min_re_vec))

   re_avg = np.mean(min_re_vec)
   print("\n{}/{} converged. \n mean min_RE of converged amount {} \n".format(n2avg-num_not_converged, n2avg,re_avg))

   file_name = directory_name+'/mean_min_re={}.npy'.format(re_avg)
   np.save(file_name,np.array(min_re_vec))

k_vec = [1,2]
bool_vec = [False, True]
network_params= {}
distribution_str = ['log-normal', 'triangular']

for _,K in enumerate(k_vec):
   for _,dropouts in enumerate(bool_vec):
      for _,third_layer in enumerate(bool_vec):
         for _,distribution in enumerate(distribution_str):

            network_params['K'] = K
            network_params['third_layer'] = third_layer
            network_params['dropouts'] = dropouts
            network_params['distribution'] = distribution

            run_one_config_of_main(network_params)


''' 
TODO:

- save the configutarion of the best epoch ! and to plot it together with the pdf !!!!
'''


