import tensorflow as tf
import numpy as np
import os
import time
import random

from RNN_GRU import RNNwavefunction


# Loading Functions --------------------------
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ To get the local energies of 2D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx*Ny) 
    - Jz: (Nx*Ny) np array
    - Bx: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx*Ny) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """
    numsamples = samples.shape[0]
    samples_reshaped = np.reshape(samples, [numsamples, Nx, Ny])

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples_reshaped[:,i]+samples_reshaped[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours)
        values = samples_reshaped[:,:,i]+samples_reshaped[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i]), axis = 1)


    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip
            valuesT[:,i][samples[:,i]==0] = 1 #Flip

            queue_samples[i+1] = valuesT

    #Calculating log_probs from samples
    #Do it in steps

    len_sigmas = (N+1)*numsamples
    steps = len_sigmas//25000+1 #I want a maximum of 25000 in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
    for j in range(numsamples):
        local_energies[j] += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,j]-0.5*log_probs_reshaped[0,j]))

    return local_energies
#--------------------------

# ---------------- Running VMC with RNNs -------------------------------------
def run_2DTFIM(numsteps = 2*10**4, systemsize_x = 5, systemsize_y = 5, Bx = +2, num_units = 50, num_layers = 1, numsamples = 500, learningrate = 1e-3, seed = 333):

    #Seeding
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    # Intitializing the RNN-----------
    units=[num_units]*num_layers#list containing the number of hidden units for each layer of the networks

    Nx=systemsize_x #x dim
    Ny=systemsize_y #y dim

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples=20 #only for initialization; later I'll use a much larger value (see below)

    wf=RNNwavefunction(Nx,Ny,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell) #contains the graph with the RNNs
    sampling=wf.sample(numsamples,input_dim) #call this function once to create the dense layers

    #now initialize everything --------------------
    with wf.graph.as_default():
        #We map a 2D configuration into a 1D configuration
        samples_placeholder=tf.placeholder(dtype=tf.int32,shape=[numsamples,Nx*Ny]) #the samples_placeholder are the samples of all of the spins
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float64,shape=[])
        learning_rate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step = global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #For exponential decay of the learning rate (only works if decay_rate < 1.0)
        probs=wf.log_probability(samples_placeholder,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer
        init=tf.global_variables_initializer()
    # End Intitializing ----------------------------

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------


    #Building the graph -------------------
    
    path=os.getcwd()

    print('Training with numsamples = ', numsamples)
    print('\n')

    Jz = +np.ones((Nx,Ny)) #Ferromagnetic couplings

    lr=np.float64(learningrate)

    ending='units'
    for u in units:
        ending+='_{0}'.format(u)
    filename='../Check_Points/2DTIM/GRU/RNNwavefunction_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending+'.ckpt'
    savename = '_2DTIM'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.float64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,Nx*Ny])
            log_probs_=wf.log_probability(samp,inputdim=2)
           
            #now calculate the fake cost function to enjoy the properties of automatic differentiation
            cost = tf.reduce_mean(tf.multiply(log_probs_,tf.stop_gradient(Eloc))) - tf.reduce_mean(tf.stop_gradient(Eloc))*tf.reduce_mean(log_probs_)

            #Calculate Gradients---------------
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            #End calculate Gradients---------------
            
            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learning_rate_withexpdecay: lr})

            saver=tf.train.Saver() #define tf saver

    #--------------------------

    meanEnergy=[]
    varEnergy=[]

    #Loading previous trainings (uncomment if you wanna restore a previous session)----------
    # print("Loading the model")
    # path=os.getcwd()
    # ending='units'
    # for u in units:
    #     ending+='_{0}'.format(u)
    # savename = '_2DTIM'
    # with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+'/'+filename)
    #         meanEnergy=np.load('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    #         varEnergy=np.load('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    #-----------

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():


            samples_ = wf.sample(numsamples=numsamples,inputdim=2)
            samples = np.ones((numsamples, Nx*Ny), dtype=np.int32)

            samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,Nx*Ny))
            log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

            queue_samples = np.zeros((Nx*Ny+1, numsamples, Nx*Ny), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
            log_probs = np.zeros((Nx*Ny+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)


            for it in range(len(meanEnergy),numsteps+1):

                print("sampling started")
                start = time.time()

                samples=sess.run(samples_)

                end = time.time()
                print("sampling ended: "+ str(end - start))

                #Estimating local_energies
                local_energies = Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                print('mean(E): {0} \pm {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))

                if it>=1000 and varE <= np.min(varEnergy): #We do it>1000 to start saving the model after we get close to convergence
                    #Saving the performances if the model is better
                    saver.save(sess,path+'/'+filename)

                if it%100==0:
                  #Saving the performances
                  np.save('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy', meanEnergy)
                  np.save('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'.npy', varEnergy)

                #lr decay
                lr_ = 1/((1/lr)+(it/10))
                #Optimization step
                sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr_})
