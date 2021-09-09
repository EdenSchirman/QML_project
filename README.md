# QML_project
This work has been done as part of a course in deep learning in Tel Aviv University. In the project, I restored and improved previous work done by IBM in quantum machine learning. Specifically, I followed Zoufal et al. [paper](https://arxiv.org/abs/1904.00043) and started with their [tutorial](https://github.com/Qiskit/qiskit-tutorials/blob/35ff38c7ffe004bf4f5f1f2e964feb4c88d32b58/tutorials/machine_learning/04_qgans_for_loading_random_distributions.ipynb). All my work is summarized in the [summary report](SummaryReport.pdf) in the repo.

The project contain several files:
1. main.py - the main script.
2. framework.py -  a script that runs several configurations of main.py
3. framework.sh - a bash script that runs framework.py in the background. To use this run in the terminal the following command (with the appropriate environement- see below, and in the folder):
```
  source framework.sh
```
4. _pytorch_discriminator_net.py - the script that the discriminator network is defined in.
5. pytorch_discriminator.py, qgan.py, quantum_generator.py - scripts that I have modified from qiskit.machine_learning package to fit my project.
6. manipulate_re_vecs.py -  ascript to extract global features of the RE vectors after they ran and already saved in the folders.

### version - conda users
For restoring the project code, please clone the git repo and then enter in terminal:
```
  conda env create -f environment.yml
```
in order to create the same environement.

### version - non-conda users
For restoring the project code, please refer to the requirements.txt file.


