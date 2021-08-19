from main import main
import numpy as np

n2avg = 10
num_not_converged = 0
min_re_vec = []
network_params= {'n_hidden0':512, 'n_hidden1': 256}
for i in range(n2avg):
   re_vec=main(network_params)  
   if re_vec[0] < re_vec[-1]:
    num_not_converged+=1
    continue
   min_re_vec.append(np.min(re_vec))

re_avg = np.mean(min_re_vec)
print("\n{}/{} converged. \n mean min_RE of converged amount {} \n".format(n2avg-num_not_converged, n2avg,re_avg))

'''TODO:
 1. try to start the next epoch with the last initial parameters 
 2. try deeper network / dropouts
 3. try conv
 '''
