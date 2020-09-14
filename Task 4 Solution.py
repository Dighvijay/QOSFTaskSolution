#!/usr/bin/env python
# coding: utf-8

# # QOSF Screening Task Solution 
# 
# #### Name - Dighvijay Giri
# #### Email ID - dighvijaygiri98@gmail dot com
# 
# Task choosen - Task 4
# 
# Find the lowest eigenvalue of the following matrix:
# 
# $$\begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & -1 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$
# 
# using VQE-like circuits, created by yourself from scratch.
# 
# <hr style="border:2px solid gray"> </hr>

# ## Overview 
# 
# 
# **VQE(Variational Quantum Eigensolver)** is a **quantum/classical hybrid** algorithm used to estimate the lowest eigenvalue of a matrix. It finds its use, for instance, in chemistry, when it is required to estimate the ground state energy of a system. It does so by finding the minimum eigenvalue of the Hermitian matrix characterizing, in this case, the molecule,
# 
# Let us borrow the formal definition from the *qiskit* textbook:<br>
# Given a Hermitian matrix $H$ with an unknown minimum eigenvalue $ \lambda_{min} $, associated with the eigenstate $ |\psi_{min}\rangle $, VQE provides an estimate $ \lambda_{\theta}$ bounding $ \lambda_{min} $:
# \begin{align*}
#     \lambda_{min} \le \lambda_{\theta} \equiv \langle \psi(\theta) |H|\psi(\theta) \rangle
# \end{align*}
# 
# where $|\psi(\theta) \rangle$
# is the eigenstate associated with $ \lambda_{\theta}$. By applying a parameterized circuit, represented by $U(\theta)$, to some arbitrary starting state $|\psi \rangle$, the algorithm obtains an estimate $U(\theta)|\psi\rangle \equiv |\psi(\theta)\rangle$ on $ |\psi_{min}\rangle $. The estimate is iteratively optimized by a classical controller changing the parameter $\theta$ minimizing the expectation value of $\langle \psi(\theta) |H|\psi(\theta) \rangle$
# 
# <hr style="border:2px solid gray"> </hr>

# The **quantum subroutine** has two fundamental steps:    
# 1. Prepare the quantum state $|\psi \rangle$, the ansatz.
# 2. Measure the expectation value $\langle \psi(\theta) |H|\psi(\theta) \rangle$. This is done by expressing $H$ as a sum of pauli matrices and performing summation over expectation values of the constituent pauli matrics.  
# 
# The variational principle ensures that this expectation value is always greater than or equal to the smallest eigenvalue of $ H $
# 
# This bound allows us to use **classical** computation to run an optimization loop to find this eigenvalue:
# 1. Use a classical non-linear optimizer to minimize the expectation value by varying ansatz parameter $\theta$. 
# 2. Iterate until convergence.
# 
# Since we will be using only one parameter, we skip the use of an optimizer and iterate over the entire range of values of $\theta$
# 
# <hr style="border:2px solid gray"> </hr>

# In[1]:


#importing dependencies
import numpy as np 
from numpy import kron

from qiskit import QuantumCircuit, Aer, execute


# ### Function to decompose given matrix into sum of pauli matrices
# 

# In[2]:


def HS(M1, M2):
    #Hilbert-Schmidt-Product of two matrices M1, M2
    return (np.dot(M1.conjugate().transpose(), M2)).trace()

def decompose(H):
    #Decompose Hermitian 4x4 matrix H into Pauli matrices
    
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    
    labels = ['I', 'X', 'Y', 'Z']
    for i in range(4):
        for j in range(4):
            label = labels[i] + labels[j]
            a_ij = 0.25 * HS(kron(S[i], S[j]), H)
            if a_ij != 0.0:
                print(a_ij,'*' ,label, '+')


# In[3]:


H = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
print("The given matrix is\n", H, "\n")
print("Matrix as a sum of pauli terms\n")
decompose(H)


# ### Preparing guess quantum state (ansatz)

# In[4]:


def prepare_ckt(angle):
    ckt = QuantumCircuit(2)
    ckt.h(0)
    ckt.cx(0, 1)
    ckt.rx(angle, 0)
    
    return ckt


# In[5]:


ckt = prepare_ckt(np.pi/2) #just an arbitrary angle for looking at the circuit
ckt.draw(output = 'mpl')


# ### Measuring Expectation Values
# 
# Starting with ZZ expectation value measurement. Creating a copy of the circuit and adding measurement elements to it.

# In[6]:


def measure_zz_circuit(ckt):
    zz_meas = ckt.copy()
    zz_meas.measure_all()
    return zz_meas

zz_meas = measure_zz_circuit(ckt)
zz_meas.draw(output ='mpl')


# In[7]:


simulator = Aer.get_backend('qasm_simulator')


# In[8]:


def measure_zz(ckt, num_shots = 10000):
    zz_meas = measure_zz_circuit(ckt)
    
    result = execute(zz_meas, backend = simulator, shots = num_shots).result()
    counts = result.get_counts(zz_meas)

    if '00' not in counts:
        counts['00'] = 0
    if '01' not in counts:
        counts['01'] = 0
    if '10' not in counts:
        counts['10'] = 0
    if '11' not in counts:
        counts['11'] = 0 

    total_counts = counts['00'] + counts['11'] + counts['01'] + counts['10']
    #signs obtained by applying zz to 00, 01, 01 and 11 states
    zz = counts['00'] + counts['11'] - counts['01'] - counts['10']  
    zz = zz / total_counts
    
    return zz


# ### To measure the expectation value of XX we need to add post rotations
# 
# Post rotations alter the ansatz to allow us to make Z pauli measurements. 

# In[9]:


#Altered Ansatz Preparation
def measure_xx_circuit(ckt):
    xx_meas = ckt.copy()
    
    xx_meas.barrier()
    xx_meas.ry(-np.pi/2,0)
    xx_meas.ry(-np.pi/2,1)
    xx_meas.measure_all()
    
    return xx_meas

xx_meas = measure_xx_circuit(ckt)
xx_meas.draw(output='mpl')


# In[10]:


def measure_xx(given_circuit, num_shots = 10000):
    
    xx_meas = measure_xx_circuit(given_circuit)
    
    result = execute(xx_meas, backend = simulator, shots = num_shots).result()
    counts = result.get_counts(xx_meas)

    if '00' not in counts:
        counts['00'] = 0
    if '01' not in counts:
        counts['01'] = 0
    if '10' not in counts:
        counts['10'] = 0
    if '11' not in counts:
        counts['11'] = 0

    total_counts = counts['00'] + counts['11'] + counts['01'] + counts['10']
    
    #signs obtained by applying zz to 00, 01, 01 and 11 states
    xx = counts['00'] + counts['11'] - counts['01'] - counts['10']
    xx = xx / total_counts
    
    return xx


# ### Similarly, to measure the expectation value of YY we need to add post rotations

# In[11]:


#Altered Ansatz Preparation
def measure_yy_circuit(ckt):
    yy_meas = ckt.copy()

    yy_meas.barrier()
    yy_meas.rx(np.pi/2, 0)
    yy_meas.rx(np.pi/2, 1)
    yy_meas.measure_all()
    
    return yy_meas

yy_meas = measure_yy_circuit(ckt)
yy_meas.draw(output='mpl')


# In[12]:


def measure_yy(given_circuit, num_shots = 10000):
    
    yy_meas = measure_yy_circuit(given_circuit)
    
    result = execute(yy_meas, backend = simulator, shots = num_shots).result()
    counts = result.get_counts(yy_meas)

    if '00' not in counts:
        counts['00'] = 0
    if '01' not in counts:
        counts['01'] = 0
    if '10' not in counts:
        counts['10'] = 0
    if '11' not in counts:
        counts['11'] = 0

    total_counts = counts['00'] + counts['11'] + counts['01'] + counts['10']
    
    #signs obtained by applying zz to 00, 01, 01 and 11 states
    yy = counts['00'] + counts['11'] - counts['01'] - counts['10']
    yy = yy / total_counts
    
    return yy


# ### Function to evaluate energy from the guess state

# In[13]:


def get_energy(given_circuit, num_shots = 10000):
    
    zz = measure_zz(given_circuit, num_shots = num_shots)
    xx = measure_xx(given_circuit, num_shots = num_shots)
    yy = measure_yy(given_circuit, num_shots = num_shots)
    
    energy = 0.5*1 + (-0.5)*xx + (-0.5)*yy + 0.5*zz 
    #coefficients obtained from decomposition into sum of Pauli matrices
    
    return energy


# ### Finding the lowest eigenvalue corresponding to the ground state energy value by iterating until convergence

# In[14]:


lowest_energy = 10000
for i in range(0, 361):
    ckt = prepare_ckt(i*np.pi/180)
    energy = get_energy(ckt) 
    
    if (lowest_energy > energy):
        lowest_energy = energy

print("The lowest eigenvalue of the given matrix is",lowest_energy)


# ### Verifying the obtained result with built-in numpy funtion 

# In[15]:


w = np.linalg.eigvals(H)
print(f'Lowest eigenvalue obtained from numpy function is {min(w)}')

assert min(w) == lowest_energy, "Results don't concide!"


# ### Conclusion
# 
# The lowest eigenvalue of the given matrix was obtained using VQE and the result was verified using built-in function. The value was found to be $\mathbf{-1.0} $ 

# ### References
# 1. __[Musty Thoughts](https://www.mustythoughts.com/variational-quantum-eigensolver-explained)__
# 2. __[Qiskit Textbook](https://qiskit.org/textbook/ch-applications/vqe-molecules.html)__
# 3. __[Grove Documentation](https://grove-docs.readthedocs.io/en/latest/vqe.html)__
