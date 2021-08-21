# genereic libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import repeat
import time
import os

# generic qiskit libraries
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.visualization import *

# QGAN functions
from qgan import QGAN
from pytorch_discriminator import PyTorchDiscriminator
from qiskit.aqua.components.neural_networks import NumPyDiscriminator

#local functions
def get_common_str(qgan):
  params = qgan.network_params
  test_str = 'K={},3rd_layer={},dropouts={},n_hidden0={},n_hidden1={},N_epochs={}'.format(params['K'], params['third_layer'], params['dropouts'], params['n_hidden0'], params['n_hidden1'], qgan._num_epochs)
  new_directory_name = 'Figures/'+ test_str
  if not os.path.exists(new_directory_name):
      os.mkdir(new_directory_name)

    
  return new_directory_name 

def plot_losses(qgan):

  dir_name = get_common_str(qgan)
  dir_name = dir_name+'/Loss'
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)

  plt.figure()
  plt.title("Loss" )
  plt.plot(qgan.epochs, qgan.g_loss, label='Generator loss function', color='mediumvioletred', linewidth=2)
  plt.plot(qgan.epochs, qgan.d_loss, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
  plt.grid()
  plt.legend(loc='best')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  if qgan.to_save:
    filename = dir_name +'/' +'_t=_'+ time.strftime("%H_%M_%S",time.localtime())+'.png'
    plt.savefig(filename)
  #plt.show()

def plot_re(qgan):
  dir_name = get_common_str(qgan)
  dir_name = dir_name+'/RE'
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)
  
  plt.figure()
  plt.title("Minimum: RE({})={}, RE({})={}".format(np.argmin(qgan.rel_entr),np.min(qgan.rel_entr),(len(qgan.rel_entr)-1), qgan.rel_entr[-1]))
  plt.plot(qgan.epochs, qgan.rel_entr, color='mediumblue', lw=4, ls=':')
  plt.xlabel('epochs')
  plt.grid()
  #plt.show()
  if qgan.to_save:
    filename = dir_name+'/' +'_t=_'+ time.strftime("%H_%M_%S",time.localtime())+'.png'
    plt.savefig(filename)
    np_filename = dir_name+'/RE_vec' +'_t=_'+ time.strftime("%H_%M_%S",time.localtime())+'.npy'
    np.save(np_filename,qgan.rel_entr)

def plot_pdf_cdf(qgan, real_data, bound):

  trunc_data = np.round(real_data)[np.round(real_data)<=bound[1]]
  generator_samples, qgan_prob = qgan.generator.get_output(qgan.quantum_instance, shots=qgan.num_samples)
  generator_samples = np.array(generator_samples)
  generator_samples = generator_samples.flatten()

  hist_data = np.histogram(trunc_data,bins=(-0.4+np.arange(2+ np.max(trunc_data))), density=True)
  classical_prob = hist_data[0]

  qgan_cdf = np.zeros(len(qgan_prob))
  classical_cdf = np.zeros(len(qgan_prob))

  curr_qgan = 0
  curr_classical =0

  for ii in range(len(qgan_prob)):
    curr_qgan += qgan_prob[ii]
    curr_classical += classical_prob[ii]

    qgan_cdf[ii] = curr_qgan
    classical_cdf[ii] = curr_classical
    

  # comparing to training data -PDF
  _, ax = plt.subplots(1,2)


  ax[0].plot(generator_samples, qgan_prob,'-o',linewidth=4, label='QGAN')
  ax[0].plot(generator_samples, classical_prob,'-o',linewidth=4, label='Classical')
  ax[0].plot(generator_samples, qgan._generator_init_prob,'-o',linewidth=2, label='QGAN initial')

  ax[0].legend(loc='best')
  ax[0].set_title('PDF')
  ax[0].grid()

  #comparing to training - CDF
  ax[1].plot(generator_samples, qgan_cdf,'-o',linewidth=4, label='QGAN')
  ax[1].plot(generator_samples, classical_cdf,'-o',linewidth=2, label='Classical')

  ax[1].legend(loc='best')
  ax[1].set_title('CDF')
  ax[1].grid()

  # dir_name, common_str = get_common_str(qgan
  dir_name = get_common_str(qgan)
  dir_name = dir_name+'/CDF_PDF'
  if not os.path.exists(dir_name):
    os.mkdir(dir_name)

  #fig.suptitle("Network parameters" + common_str)
  #plt.show()
  if qgan.to_save:
    filename = dir_name+'/' + '_t=_'+ time.strftime("%H_%M_%S",time.localtime())+'.png'
    plt.savefig(filename)

def plot_training_data(real_data, bound):
  fig, ax = plt.subplots(1,2)

  rounded_data = np.round(real_data)
  ax[0].hist(rounded_data,bins=np.arange(np.max(rounded_data)))
  ax[0].grid()

  trunc_data = rounded_data[rounded_data<=bound[1]]
  num_bins = np.int(1+ np.max(trunc_data))
  n = ax[1].hist(trunc_data,bins=(-0.4+np.arange(2+ np.max(trunc_data))), density=True, width=0.8)
  ax[1].grid()

  probabilities = n[0]
  # values = 0.5 + np.arange(1 + np.max(trunc_data))
  # ax[1].plot(values, probabilities,'o',linewidth=4)
  # plt.show()

  np.set_printoptions(precision=2)
  print(probabilities)

def initialize_default_params(network_params):
  if 'lr' not in network_params.keys():
    network_params['lr'] = 1e-4
  if 'is_amsgrad' not in network_params.keys():
    network_params['is_amsgrad'] = True
  if 'n_hidden0' not in network_params.keys():
    network_params['n_hidden0'] = int(512)
  if 'n_hidden1' not in network_params.keys():
    network_params['n_hidden1'] = int(256)
  if 'include_bias' not in network_params.keys():
    network_params['include_bias'] = True
  if 'dropouts' not in network_params.keys():
    network_params['dropouts'] = False
  if 'conv_net' not in network_params.keys():
    network_params['conv_net'] = False
  if 'seed' not in network_params.keys():
    network_params['seed'] = 71
  if 'third_layer' not in network_params.keys():
    network_params['third_layer']= False
  if 'K' not in network_params.keys():
    network_params['K']= 2

  return network_params

def main(network_params: dict={}):
  network_params = initialize_default_params(network_params)
  # parameters difference between 2 and 3 qubits
  num_qubits = [3]
  to_plot = True
  seed = network_params['seed']
  K = network_params['K'] # reps - how many entanglement layers 
  np.random.seed(seed)
  aqua_globals.random_seed = seed

  if num_qubits[0]==2:
    # training parameters
    N=1000
    num_epochs = 10 
    batch_size = 500 #500

    init_params = [3., 1., 0.6, 1.6] # the parameters are the initial rotation angles around the Y axis.
    entangler_map = [[0, 1]]
    
  elif num_qubits[0]==3:
    # training parameters
    num_epochs = 500  
    batch_size = 1000
    N=5000

    if 'params_g' in network_params.keys():
      init_params = network_params['params_g']
    else:
      # because the QuantumGenerator initializaton takes the parameters close to zero, they might get stuck on a local minimum
      init_params = 2*np.pi*np.random.rand((K+1)*num_qubits[0]) *2e-2
      init_params = init_params.flatten()
    entangler_map= 'full'

  # log-normal distrbuition parameters
  mu = 1
  sigma = 1

  # extracting training data
  real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)

  # quantum parameters
  max_state = 2**num_qubits[0] -1 
  bounds = np.array([0, max_state]) 
  num_registers = 1

  # understanding the training data
  # plot_training_data(real_data, bound)

  # initialize QGAN
  # https://qiskit.org/documentation/stubs/qiskit.aqua.algorithms.QGAN.html
  qgan = QGAN(real_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)
  qgan.seed = 1

  # set quantum instance to run quantum generator
  quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                    seed_transpiler=seed, seed_simulator=seed)


  # Set an initial state for the generator circuit
  init_dist = UniformDistribution(sum(num_qubits))
  var_form = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz', reps=K, entanglement=entangler_map) #

  # Set generator circuit by adding the initial distribution infront of the ansatz
  g_circuit = var_form.compose(init_dist, front=True)

  # Set quantum generator
  qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)

  # The parameters have an order issue that following is a temp. workaround
  qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)

  discriminator = PyTorchDiscriminator(network_params)
  qgan.set_discriminator(discriminator)

  # Run qGAN
  qgan.debug = True
  qgan.to_save = True
  result = qgan.run(quantum_instance)
  # print(result)

  # analyzing the results
  qgan.epochs = np.arange(num_epochs)
  qgan.network_params = network_params
  qgan.num_samples = N
  #params_g = result['params_g']
  
  #plots
  if to_plot:
    plot_losses(qgan)
    plot_re(qgan) # The losses are not good measures for evaluating GAN. RE is exactly what we re looking to minimize !
    plot_pdf_cdf(qgan, real_data, bounds)
  
  new_directory_name = get_common_str(qgan)
  return qgan.rel_entr, new_directory_name

if __name__ =="__main__":
  main()
  plt.show()





