import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
from math import ceil

from RNNwavefunction import RNNwavefunction
from MDRNNcell import MDRNNcell

# Loading Functions --------------------------
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ To get the local energies of 2D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny) np array
    - Bx: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx,Ny) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """

    numsamples = samples.shape[0]

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice :)))
        values = samples[:,:,i]+samples[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i]), axis = 1)


    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(Nx):  #Non-diagonal elements
            for j in range(Ny):
                valuesT = np.copy(samples)
                valuesT[:,i,j][samples[:,i,j]==1] = 0 #Flip
                valuesT[:,i,j][samples[:,i,j]==0] = 1 #Flip

                queue_samples[i*Ny+j+1] = valuesT

    #Calculating log_probs from samples
    #Do it in steps

    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/25000) #Get a maximum of 25000 configurations in batch size to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, Nx,Ny])
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})
      # print(i)

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
    for j in range(numsamples):
        local_energies[j] += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,j]-0.5*log_probs_reshaped[0,j]))

    return local_energies
#--------------------------


# ---------------- Running VMC with 2DRNNs -------------------------------------
def run_2DTFIM(numsteps = 2*10**4, systemsize_x = 5, systemsize_y = 5, Bx = +2, num_units = 50, numsamples = 500, learningrate = 5e-3, seed = 111):

    #Seeding
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    # Intitializing the RNN-----------
    units=[num_units] #list containing the number of hidden units for each layer of the networks (We only support one layer for the moment)
     
    Nx=systemsize_x #x dim
    Ny=systemsize_y #y dim

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #number of samples only for initialization
    wf=RNNwavefunction(Nx,Ny,units=units,cell=MDRNNcell,seed = seed) #contains the graph with the RNNs

    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers

    #now initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder=tf.placeholder(dtype=tf.int32,shape=[numsamples_,Nx,Ny]) #the samples_placeholder are the samples of all of the spins
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float64,shape=[])
        learning_rate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step = global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #For exponential decay of the learning rate (only works if decay_rate < 1.0)
        probs=wf.log_probability(samples_placeholder,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer
        init=tf.global_variables_initializer()
    # End Intitializing

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------

    with wf.graph.as_default():
        variables_names =[v.name for v in tf.trainable_variables()]
    #     print(variables_names)
        sum = 0
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            v1 = tf.reshape(v,[-1])
            print(k,v1.shape)
            sum +=v1.shape[0]
        print('The sum of params is {0}'.format(sum))


    meanEnergy=[]
    varEnergy=[]

    #Running the training -------------------
    path=os.getcwd()

    print('Training with numsamples = ', numsamples)
    print('\n')

    Jz = +np.ones((Nx,Ny))

    lr=np.float64(learningrate)
    ending='units'
    for u in units:
        ending+='_{0}'.format(u)
    filename='../Check_Points/2DTFIM/RNNwavefunction_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending+'.ckpt'
    savename = '_2DTFIM'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.float64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,Nx,Ny])
            log_probs_=wf.log_probability(samp,inputdim=2)

            cost = tf.reduce_mean(tf.multiply(log_probs_,tf.stop_gradient(Eloc))) - tf.reduce_mean(tf.stop_gradient(Eloc))*tf.reduce_mean(log_probs_)

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learning_rate_withexpdecay: lr})

            saver=tf.train.Saver() #define tf saver

    ##Loading previous trainings - Uncomment if you want to load the model----------
    # print("Loading the model")
    # with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+'/'+filename)
    #         meanEnergy=np.load('../Check_Points/2DTFIM/meanEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    #         varEnergy=np.load('../Check_Points/2DTFIM/varEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    #-----------


    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():

            samples_ = wf.sample(numsamples=numsamples,inputdim=2)
            samples = np.ones((numsamples, Nx,Ny), dtype=np.int32)

            samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,Nx,Ny))
            log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

            queue_samples = np.zeros((Nx*Ny+1, numsamples, Nx,Ny), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
            log_probs = np.zeros((Nx*Ny+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

            for it in range(len(meanEnergy),numsteps+1):

#                 print("sampling started")
#                 start = time.time()
                samples=sess.run(samples_)
#                 end = time.time()
#                 print("sampling ended: "+ str(end - start))

                #Estimating local_energies
                local_energies = Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                if it%10==0:                
                   print('mean(E): {0}, var(E): {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))
              
            #Comment if you dont want to save or if saving is not working
                if it>=5000 and varE <= np.min(varEnergy): #5000 can be changed to suite your chosen number of iterations and to avoid slow down by saving the model too often during the initial phase of fast convergence
                  #Saving the performances if the model is better
                  saver.save(sess,path+'/'+filename)

            #Comment if you dont want to save or if saving is not working
                if it%10==0:
                  #Saving the performances
                  np.save('../Check_Points/2DTFIM/meanEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy', meanEnergy)
                  np.save('../Check_Points/2DTFIM/varEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'.npy', varEnergy)

                #lr_adaptation
                lr_ = lr*(1+it/5000)**(-1)
                #Optimize
                sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr_})
    return meanEnergy, varEnergy
