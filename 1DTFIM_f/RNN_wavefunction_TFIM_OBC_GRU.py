import tensorflow as tf
import numpy as np
import os
import time
import random

from RNN_GRU import RNNwavefunction
# from RNN_GRU_par import RNNwavefunction #To use an RNN that has a parity symmetry (but comment the first)

# Loading Functions --------------------------
def Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ To get the local energies of 1D TFIM (OBC) given a set of set of samples in parallel!"""
    numsamples = samples.shape[0]
    N = samples.shape[1]

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1): #diagonal elements (let's do 1D for simple stuff)
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += valuesT*(-Jz[i])

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip
            valuesT[:,i][samples[:,i]==0] = 1 #Flip

            queue_samples[i+1] = valuesT

    #Calculating log_probs from samples
    #Do it in steps

    # print("Estimating log probs started")
    # start = time.time()

    len_sigmas = (N+1)*numsamples
    steps = len_sigmas//25000+1 #I want a maximum of 25000 in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    # end = time.time()
    # print("Estimating log probs ended ", end-start)

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
    for j in range(numsamples):
        local_energies[j] += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,j]-0.5*log_probs_reshaped[0,j]))

    return local_energies
#--------------------------

# ---------------- Running VMC with RNNs -------------------------------------
def run_1DTFIM(numsteps = 10**4, systemsize = 20, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 5e-3, seed = 111):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    #End Seeding ---------------------------------------------

    # System size
    N = systemsize

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples=20 #only for initialization; later I'll use a much larger value (see below)

    wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    sampling=wf.sample(numsamples,input_dim) #call this function once to create the dense layers

    #now initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder=tf.placeholder(dtype=tf.int32,shape=[numsamples,N]) #the samples_placeholder are the samples of all of the spins
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

    #Counting the number of parameters
    with wf.graph.as_default():
        variables_names =[v.name for v in tf.trainable_variables()]
    #     print(variables_names)
        sum = 0
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            v1 = tf.reshape(v,[-1])
            print(k,v1.shape)
            sum +=v1.shape[0]
        print('The number of params is {0}'.format(sum))

    #Building the graph -------------------

    path=os.getcwd()

    Jz = +np.ones(N) #Ferro coupling
    Bx = +1 #Critical Point

    #for a given network, generate a large number of samples:
    numsamples = num_samples

    #Learning rate
    lr=np.float64(learningrate)

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)

    filename='/../Check_Points/TIM/GRU/RNNwavefunction_N'+str(N)+'_samp'+str(numsamples)+'_Jz1Bx1_GRURNN_OBC'+ending + '.ckpt'
    savename = '_TIM'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.float64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_probs_=wf.log_probability(samp,inputdim=2)

            #now calculate the fake cost function to enjoy the properties of automatic differentiation
            cost = tf.reduce_mean(tf.multiply(log_probs_,Eloc)) - tf.reduce_mean(Eloc)*tf.reduce_mean(log_probs_) #factor of 2 in the above equation

            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)
            sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learningrate_placeholder: lr})
            saver=tf.train.Saver()
    #----------------------------------------------------------------

    meanEnergy=[]
    varEnergy=[]

    #Loading previous trainings (uncomment if you wanna restore a previous session)----------
    # path=os.getcwd()
    # ending='_units'
    # for u in units:
    #     ending+='_{0}'.format(u)
    # savename = '_TIM'
    # with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+filename)
    #         meanEnergy=np.load('../Check_Points/TIM/GRU/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy').tolist()
    #         varEnergy=np.load('../Check_Points/TIM/GRU/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy').tolist()
    #------------------------------------

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():


          samples_ = wf.sample(numsamples=numsamples,inputdim=2)
          samples = np.ones((numsamples, N), dtype=np.int32)

          samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
          log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

          queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
          log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)


          for it in range(len(meanEnergy),numsteps+1):

              print("sampling started")
              start = time.time()

              samples=sess.run(samples_)

              end = time.time()
              print("sampling ended: "+ str(end - start))

              #Estimating local_energies
              local_energies = Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess)

              meanE = np.mean(local_energies)
              varE = np.var(local_energies)

              #adding elements to be saved
              meanEnergy.append(meanE)
              varEnergy.append(varE)

              if it%1==0:
                  print('mean(E): {0} \pm {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))

              sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr})

              if it%50==0 and varE <= np.min(varEnergy[::50]):
                  #Saving the performances if the model is better
                  saver.save(sess,path+'/'+filename)

              if it%100==0:
                  #Saving the performances
                  np.save('../Check_Points/TIM/GRU/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',meanEnergy)
                  np.save('../Check_Points/TIM/GRU/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',varEnergy)

              if it%100==0:
                  print("Learning rate = " + str(sess.run(learning_rate_withexpdecay,feed_dict={learningrate_placeholder: lr}))+ "\n\n")

    #----------------------------
