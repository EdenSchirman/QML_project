# genereic libraries
import matplotlib.pyplot as plt
import numpy as np

# generic qiskit libraries
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import TwoLocal, UniformDistribution
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.visualization import *

# QGAN functions
from qiskit.aqua.algorithms import QGAN
from qiskit.aqua.components.neural_networks import NumPyDiscriminator
from qiskit_machine_learning.algorithms.distribution_learners.qgan import PyTorchDiscriminator
# from qiskit_machine_learning.algorithms.distribution_learners.qgan import pytorch_discriminator


def plot_losses(qgan):
  # epochs = qgan.epochs
  plt.figure()
  plt.title("Progress in the loss function")
  plt.plot(qgan.epochs, qgan.g_loss, label='Generator loss function', color='mediumvioletred', linewidth=2)
  plt.plot(qgan.epochs, qgan.d_loss, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
  plt.grid()
  plt.legend(loc='best')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  #plt.show()

def plot_re(qgan):
  plt.figure()
  plt.title("Relative Entropy. Network parameters:[{},{},{},{}], bias={} ".format(qgan.n_in,
    qgan.n_hidden0, qgan.n_hidden1, qgan.n_out, qgan.include_bias))
  plt.plot(qgan.epochs, qgan.rel_entr, color='mediumblue', lw=4, ls=':')
  plt.xlabel('epochs')
  plt.grid()
  #plt.show()
  filename = 'Figures/RE, [{},{},{},{}], bias={}.png'.format(qgan.n_in,
    qgan.n_hidden0, qgan.n_hidden1, qgan.n_out, qgan.include_bias)
  plt.savefig(filename)

def plot_pdf_cdf(qgan, real_data, bound):

  trunc_data = np.round(real_data)[np.round(real_data)<=bound[1]]
  generator_samples, qgan_prob = qgan.generator.get_output(qgan.quantum_instance, shots=N)
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
  fig, ax = plt.subplots(1,2)

  ax[0].plot(generator_samples, qgan_prob,'-o',linewidth=4, label='QGAN')
  ax[0].plot(generator_samples, classical_prob,'-o',linewidth=4, label='Classical')

  ax[0].legend(loc='best')
  ax[0].set_title('PDF')
  ax[0].grid()

  #comparing to training - CDF
  ax[1].plot(generator_samples, qgan_cdf,'-o',linewidth=4, label='QGAN')
  ax[1].plot(generator_samples, classical_cdf,'-o',linewidth=4, label='Classical')

  ax[1].legend(loc='best')
  ax[1].set_title('CDF')
  ax[1].grid()

  fig.suptitle("Network parameters:[{},{},{},{}], bias={} ".format(qgan.n_in,
    qgan.n_hidden0, qgan.n_hidden1, qgan.n_out, qgan.include_bias))
  #plt.show()
  filename = "Figures/PDF_CDF, [{},{},{},{}], bias={}.png".format(qgan.n_in,
  qgan.n_hidden0, qgan.n_hidden1, qgan.n_out, qgan.include_bias)
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


to_plot = True
seed = 71
np.random.seed = seed
aqua_globals.random_seed = seed

# log-normal distrbuition parameters
N = 1000
mu = 1
sigma = 1

# extracting training data
real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)

# quantum parameters
num_qubits = [2] 
max_num_qubits = np.max(num_qubits)
max_num_qubits = max_num_qubits.flatten()
max_state = 2**max_num_qubits[0] -1 
bound = np.array([0., max_state]) 
num_registers = len(num_qubits)

# training parameters
num_epochs = 10 
batch_size = 100

# understanding the training data
# plot_training_data(real_data, bound)

# initialize QGAN
# https://qiskit.org/documentation/stubs/qiskit.aqua.algorithms.QGAN.html
qgan = QGAN(real_data, bound, num_qubits, batch_size, num_epochs, snapshot_dir=None)
qgan.seed = 1

# set quantum instance to run quantum generator
quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                   seed_transpiler=seed, seed_simulator=seed)

# which qubits would be entangled to which
entangler_map = [[0, 1]]

# Set an initial state for the generator circuit
init_dist = UniformDistribution(sum(num_qubits))

# Set the ansatz circuit - the part of the circuit that transforms psi_in into the desired distribuition
# reps - how many entanglement layers 
var_form = TwoLocal(int(np.sum(num_qubits)), 'ry', 'cz',entanglement=entangler_map, reps=1) #

# the parameters are the initial rotation angles around the Y axis.
init_params = [3., 1., 0.6, 1.6]
# init_params = np.random.rand(var_form.num_parameters_settable) * 2 * np.pi

# Set generator circuit by adding the initial distribution infront of the ansatz
g_circuit = var_form.compose(init_dist, front=True)
print('The paramatrized circuit: \n',g_circuit)

# Set quantum generator
qgan.set_generator(generator_circuit=g_circuit, generator_init_params=init_params)

# The parameters have an order issue that following is a temp. workaround
qgan._generator._free_parameters = sorted(g_circuit.parameters, key=lambda p: p.name)

# Set classical discriminator neural network
# discriminator = NumPyDiscriminator(len(num_qubits))
qgan.n_in = 1
qgan.n_out = 1
qgan.n_hidden0 = 512
qgan.n_hidden1 = 256
qgan.include_bias = False


discrminator = PyTorchDiscriminator(qgan.n_in, qgan.n_out, qgan.n_hidden0, qgan.n_hidden1, qgan.include_bias)
qgan.set_discriminator(discrminator)

# Run qGAN
result = qgan.run(quantum_instance)
print(result)

# analyzing the results
qgan.epochs = np.arange(num_epochs)

#plots
if to_plot:
  # plot_losses(qgan)
  plot_re(qgan) # The losses are not good measures for evaluating GAN. RE is exactly what we re looking to minimize !
  plot_pdf_cdf(qgan, real_data, bound)

