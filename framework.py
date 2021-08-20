from main import main
import numpy as np

# parameters
n2avg = 5
num_not_converged = 0
min_re_vec = []
network_params= {'n_hidden0':512, 'n_hidden1': 256, 'conv_net':True, 'droputs': False}
seed = [767, 3, 13, 94, 123]

# running main
for i in range(n2avg):
   network_params['seed']=seed[i]
   re_vec, _ = main(network_params)  
   if re_vec[0] + 0.03 < re_vec[-1]:
    num_not_converged+=1
    continue
   # network_params['params_g'] = params_g
   min_re_vec.append(np.min(re_vec))

   print('\n ***************************** \n Finished run {}/{} \n rel_entropy = {} \n **************************** \n'.format(i+1, n2avg, min_re_vec))

re_avg = np.mean(min_re_vec)
print("\n{}/{} converged. \n mean min_RE of converged amount {} \n".format(n2avg-num_not_converged, n2avg,re_avg))

'''TODO:
 1. Done - try to start the next epoch with the last initial parameters 
 2. try deeper network / dropouts
 3. try conv
 3. Done - check if D(g^l/ x^l) return a in [0,1] or {0,1} . so it is all about working with the probabilities to recieve each value.
 
 '''
