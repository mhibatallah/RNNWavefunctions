import tensorflow as tf
import numpy as np
import os
import time
import random

from RNN_2D_GRU import RNNwavefunction

import sys
ID = int(sys.argv[1]) #Getting the unit number from the array

hs = [2,3,4]
list_units = [200]
num_layers = 3

import itertools
tasks = list(itertools.product(list_units, hs))

parameters = tasks[ID]

test = 1
num_units = parameters[0]
h = parameters[1]

print(test, num_units, h)

#Seeding
tf.reset_default_graph()
random.seed(111)  # `python` built-in pseudo-random generator
np.random.seed(111)  # numpy pseudo-random generator
tf.set_random_seed(111)  # tensorflow pseudo-random generator

# Intitializing the RNN-----------
units=[num_units]*num_layers#list containing the number of hidden units for each layer of the networks

Nx=12 #x dim
Ny=12 #y dim

input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
numsamples=20 #only for initialization; later I'll use a much larger value (see below)
#cell=tf.contrib.rnn.LSTMCell()
# wf=RNNwavefunction(Nx,Ny,units=units,cell=tf.contrib.rnn.BasicRNNCell) #contains the graph with the RNNs
wf=RNNwavefunction(Nx,Ny,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell) #contains the graph with the RNNs
sampling=wf.sample(numsamples,input_dim) #call this function once to create the dense layers

with wf.graph.as_default(): #now initialize everything
    inputs=tf.placeholder(dtype=tf.int32,shape=[numsamples,Nx*Ny]) #the inputs are the samples of all of the spins

    #defining adaptive learning rate
    global_step = tf.Variable(0, trainable=False)
    learningrate=tf.placeholder(dtype=tf.float64,shape=[])
    learning_rate = tf.train.exponential_decay(learningrate, global_step, 200, 1.00, staircase=True) #decay every 10 step
    probs=wf.log_probability(inputs,input_dim) #The probs are obtained by feeding the sample of spins.
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    init=tf.global_variables_initializer()
# End Intitializing

#Starting Session------------
#Activating GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess=tf.Session(graph=wf.graph, config=config)
sess.run(init)
#---------------------------

# Loading Functions --------------------------

def IsingMatrixElements(Jz,Bx,sigmap,sigmas, matrixelements):
    """
    computes the matrix element of the open TIM Hamiltonian for a given state sigmap
    -----------------------------------------------------------------------------------
    Parameters:
    Jz: np.ndarray of shape (N), respectively, and dtype=float:
                Ising parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
                A sample of spins can be fed here.
    Bx: Transvers magnetic field (N)
    -----------------------------------------------------------------------------------
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
             sigmas:         np.ndarray of dtype=int and shape (?,N)
                             the states for which there exist non-zero matrix elements for given sigmap
             matrixelements: np.ndarray of dtype=float and shape (?)
                             the non-zero matrix elements
    """
    #the diagonal part is simply the sum of all Sz-Sz interactions
    diag=0
    num = 0 #Number of basis elements


    ### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
    N_spins = len(Jz)

    neighbours = np.zeros((N_spins,2),dtype=np.int32)
    for i in range(N_spins):
    #neighbour to the right:
      neighbours[i,0]=i+1
      # if i%Nx==(Nx-1):
      #   neighbours[i,0]=i+1-Nx
    #upwards neighbour:
      neighbours[i,1]=i+Nx
      # if i >= (N_spins-Ny):
      #   neighbours[i,1]=i+Ny-N_spins

    for ny in range(Ny):
        for nx in range(Nx):
            i = ny*Nx + nx
            if nx != (Nx-1):  #if not on the right
                if sigmap[i] != sigmap[neighbours[i,0]]:
                    diag-=0.25*Jz[i] #add a negative energy contribution
                else:
                    diag+=0.25*Jz[i]

            if ny != (Ny-1): #if not on the up
                if sigmap[i] != sigmap[neighbours[i,1]]:
                    diag-=0.25*Jz[i] #add a negative energy contribution
                else:
                    diag+=0.25*Jz[i]


    matrixelements[num] = diag #add the diagonal part to the matrix elements
    sig = np.copy(sigmap)

    sigmas[num] = sig
    # sigmas_dec[num] = int(''.join(str(s) for s in sig),2)

    num += 1

    #off-diagonal part (For the transverse Ising Model)
    for site in range(N_spins):
      sig = np.copy(sigmap)
      if sig[site] == 1:
          sig[site] = 0
      else:
          sig[site] = 1

      sigmas[num] = sig
      # sigmas_dec[num] = int(''.join(str(s) for s in sig),2)
      matrixelements[num] = Bx[site]/2

      num += 1
    return num

def IsingLocalEnergies(Jz,Bx,sigmasp,sigmas, H, sigmaH, matrixelements):
    """
    DEPRECATED
    computes the local energy for the Ising model:
    ---------------------------------------------------------------------------------
    Parameters:
    Jz: np.ndarray of shape (N), respectively, and dtype=float:
                Ising parameters
    sigmasp:    np.ndarrray of dtype=int and shape (numsamples,N)
                spin-states, integer encoded (using 0 for down spin and 1 for up spin)
    Bx: Transvers magnetic field (N)
    ----------------------------------------------------------------------------------
    """
    slices=[]
    sigmas_length = 0

    print("Generating Matrix Elements Started")
    start = time.time()

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = IsingMatrixElements(Jz, Bx,sigmap, sigmaH, matrixelements)#note that sigmas[0,:]==sigmap
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        if (len(H[s])!=num):
            print("error")
            print(H[s].shape,s, matrixelements[:num].shape)

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    end = time.time()
    print("Generating Matrix Elements Ended : " + str(end-start))

    return slices
#--------------------------

meanEnergy=[]
varEnergy=[]

#Running the training -------------------
numsamples = 500
numsteps = 50000
lr_ = 1e-3
path=os.getcwd()

print('Training with numsamples = ', numsamples)
print('\n')

Jz = -4*np.ones(Nx*Ny)
Bx = -2*h*np.ones(Nx*Ny)

lr=np.float64(lr_)
ending='units'
for u in units:
    ending+='_{0}'.format(u)
filename='../Check_Points/2DTIM/GRU/RNNwavefunction_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending+'_test'+str(test)+'.ckpt'
savename = '_2DTIM'

with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    with wf.graph.as_default():
        Eloc=tf.placeholder(dtype=tf.float64,shape=[numsamples])
        samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,Nx*Ny])
#         basis_ = tf.placeholder(dtype=tf.int32,shape=[2**N,N])
        log_probs_=wf.log_probability(samp,inputdim=2)
        #test=sess.run(log_probs_,feed_dict={samp:samples})
        #print(np.linalg.norm(test-log_probs))
        #now calculate the fake cost function:
        cost = tf.reduce_mean(tf.multiply(log_probs_,tf.stop_gradient(Eloc))) - tf.reduce_mean(tf.stop_gradient(Eloc))*tf.reduce_mean(log_probs_) #factor of 2 in the above equation
        #print(sess.run(cost,feed_dict={Eloc:-local_energies}))
                                          #cancels when taking log(sqrt(prob))=log(sqrt(psi^2))`
                                          #=log(psi)=2*log(psi^2)->log(psi)=1/2*log(psi^2)=1/2*log_probs
        gradients, variables = zip(*optimizer.compute_gradients(cost))
        #clipped_gradients,_=tf.clip_by_global_norm(gradients,1.0)
        optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
        sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learningrate: lr})

        saver=tf.train.Saver() #define tf saver

#Loading previous trainings----------

# print("Loading the model")
# numsamples = 500
# path=os.getcwd()
# ending='units'
# for u in units:
#     ending+='_{0}'.format(u)
# filename='../Check_Points/2DTIM/GRU/RNNwavefunction_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending+'_test'+str(test)+'.ckpt'
# savename = '_2DTIM'
# with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
#     with wf.graph.as_default():
#         saver.restore(sess,path+'/'+filename)
#         meanEnergy=np.load('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy').tolist()
#         varEnergy=np.load('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy').tolist()
#-----------


with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    with wf.graph.as_default():
      # max_grad = tf.reduce_max(tf.abs(gradients[0]))

      N = Nx*Ny
      samples_ = wf.sample(numsamples=numsamples,inputdim=2)
      samples = np.ones((numsamples, N), dtype=np.int32)

      inputs=tf.placeholder(dtype=tf.int32,shape=(None,N))
      log_probs=wf.log_probability(inputs,inputdim=2)

      local_energies = np.zeros(numsamples, dtype = np.float64) #The type complex should be specified, otherwise the imaginary part will be discarded

      sigmas=np.zeros(((N+1)*numsamples,N), dtype=np.int32)
      H = np.zeros((N+1)*numsamples, dtype=np.float64)
      log_probabilities = np.zeros((N+1)*numsamples, dtype=np.float64)

      sigmaH = np.zeros((N+1,N), dtype = np.int32)
      matrixelements=np.zeros(N+1, dtype = np.float64)

      for it in range(len(meanEnergy),numsteps+1):
        print("sampling started")
        start = time.time()

        samples=sess.run(samples_)

        end = time.time()
        print("sampling ended: "+ str(end - start))

        slices = IsingLocalEnergies(Jz,Bx,samples, sigmas, H, sigmaH, matrixelements)

        #Getting the unique sigmas with the matrix elements
        #Process in steps to get log probs
        print("Generating log amplitudes Started")
        start = time.time()
        len_sigmas = (N+1)*numsamples
        steps = len_sigmas//50000+1

        print("number of required steps :" + str(steps))

        for i in range(steps):
            if i < steps-1:
                cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
            else:
                cut = slice((i*len_sigmas)//steps,len_sigmas)

            log_probabilities[cut] = sess.run(log_probs,feed_dict={inputs:sigmas[cut]})
            print(i+1)

        end = time.time()
        print("Generating log amplitudes ended "+ str(end - start))

        #Generating the local energies
        for n in range(len(slices)):
            s=slices[n]
            local_energies[n] = H[s].dot(np.exp(0.5*log_probabilities[s]-0.5*log_probabilities[s][0]))

        meanE = np.mean(local_energies)
        varE = np.var(local_energies)

        #adding elements to be saved
        meanEnergy.append(meanE)
        varEnergy.append(varE)

        print('mean(E): {0} \pm {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))

        #lr_adap
        lr_ = 1/((1/lr)+(it/10))

        sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate: lr_})
        # print(sess.run(max_grad,feed_dict={Eloc:local_energies,samp:samples}))

        if it%500==0 and varE <= np.min(varEnergy):
          #Saving the performances if the model is better
          saver.save(sess,path+'/'+filename)

        if it%100==0:
          #Saving the performances
          np.save('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy', meanEnergy)
          np.save('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'_test'+str(test)+'.npy', varEnergy)


        if it%200==0:
          print("Learning rate = " + str(sess.run(learning_rate,feed_dict={learningrate: lr}))+ "\n\n")
