import tensorflow as tf
import numpy as np
import os
import time
import random

from ComplexRNN_GRU import RNNwavefunction

#Loading functions:
# Loading Functions --------------------------
def J1J2MatrixElements(J1,J2,Bz,sigmap, sigmas, matrixelements, periodic = False, Marshall_sign = False):
    """
    computes the matrix element of the J1J2 model for a given state sigmap
    -----------------------------------------------------------------------------------
    Parameters:
    J1, J2, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                J1J2 parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
                A sample of spins can be fed here.
    -----------------------------------------------------------------------------------
    Returns: ...
    """
    N=len(Bz)
    #the diagonal part is simply the sum of all Sz-Sz interactions plus a B field
    diag=np.dot(sigmap-0.5,Bz)

    num = 0 #Number of basis elements

    if periodic:
        limit = N
    else:
        limit = N-1

    for site in range(limit):
        if sigmap[site]!=sigmap[(site+1)%N]: #if the two neighouring spins are opposite
            diag-=0.25*J1[site] #add a negative energy contribution
        else:
            diag+=0.25*J1[site]
        if site<(limit-1) and J2[site] != 0.0:
            if sigmap[site]!=sigmap[(site+2)%N]: #if the two second neighouring spins are opposite
                diag-=0.25*J2[site] #add a negative energy contribution
            else:
                diag+=0.25*J2[site]

    matrixelements[num] = diag #add the diagonal part to the matrix elements

    sig = np.copy(sigmap)

    sigmas[num] = sig

    num += 1

    #off-diagonal part:
    for site in range(limit):
        if J1[site] != 0.0:
          if sigmap[site]!=sigmap[(site+1)%N]:
              sig=np.copy(sigmap)
              sig[site]=sig[(site+1)%N] #Make the two neighbouring spins equal.
              sig[(site+1)%N]=sigmap[site]

              sigmas[num] = sig #The last fours lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)

              if Marshall_sign:
                  matrixelements[num] = -J1[site]/2
              else:
                  matrixelements[num] = +J1[site]/2

              num += 1

    for site in range(limit-1):
      if J2[site] != 0.0:
        if sigmap[site]!=sigmap[(site+2)%N]:
            sig=np.copy(sigmap)
            sig[site]=sig[(site+2)%N] #Make the two neighbouring spins equal.
            sig[(site+2)%N]=sigmap[site]

            sigmas[num] = sig #The last fours lines are meant to flip the two neighbouring spins (that the effect of applying J+ and J-)
            matrixelements[num] = +J2[site]/2

            num += 1
    return num

def J1J2Slices(J1, J2, Bz, sigmasp, sigmas, H, sigmaH, matrixelements):
    """
    computes the local energies for the open J1J2 model for a given spin-state sample sigmasp:
    Eloc(\sigma')=\sum_{sigma} H_{\sigma'\sigma}\psi_{\sigma}/\psi_{\sigma'}
    ----------------------------------------------------------------------------
    Parameters:
    Jz, Jxy, Bz, Bx: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                XXZ parameters
    sigmasp:    np.ndarrray of dtype=int and shape (numsamples,N)
                spin-states, integer encoded (using 0 for down spin and 1 for up spin)
    RNN:        fully initialized RNNwavefunction object
    ----------------------------------------------------------------------------
    """

    slices=[]
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements)#note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length
#--------------------------

# ---------------- Running VMC with RNNs for J1J2 Model -------------------------------------
def run_J1J2(numsteps = 10**5, systemsize = 20, J1_  = 1.0, J2_ = 0.0, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 2.5*1e-4, seed = 111):

    N=systemsize #Number of spins
    lr = np.float64(learningrate)

    #Seeding
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator


    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples=20 #only for initialization; later I'll use a much larger value (see below)
    wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
     #contains the graph with the RNNs
    sampling=wf.sample(numsamples,input_dim) #call this function once to create the dense layers


    with wf.graph.as_default(): #now initialize everything
        inputs=tf.placeholder(dtype=tf.int32,shape=[numsamples,N]) #the inputs are the samples of all of the spins
        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float32,shape=[])
        learningrate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #Adaptive Learning (decay_rate = 1 -> no decay)
        amplitudes=wf.log_amplitude(inputs,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learningrate_withexpdecay, beta1=0.9, beta2 = 0.999, epsilon = 1e-8)
        init=tf.global_variables_initializer()
    # End Intitializing

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------


    #Running the training -------------------

    path=os.getcwd()

    J1=+J1_*np.ones(N)
    J2=+J2_*np.ones(N)

    Bz=+0.0*np.ones(N)

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)

    numsamples = 500

    savename = '_J1J2'+str(J2[0])

    filename='/../Check_Points/J1J2/RNNwavefunction_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.ckpt'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.complex64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_amplitudes_=wf.log_amplitude(samp,inputdim=2)

            #now calculate the fake cost function: https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
            cost = 2*tf.real(tf.reduce_mean(tf.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()),feed_dict={learningrate_placeholder: lr})

            saver=tf.train.Saver() #define tf saver


    meanEnergy=[]
    varEnergy=[]

    # #Loading previous trainings----------
    # with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+''+filename)
    #         meanEnergy = np.load('../Check_Points/J1J2/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy').tolist()
    #         varEnergy = np.load('../Check_Points/J1J2/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy').tolist()
    ## -----------
    #Running The training

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
          # max_grad = tf.reduce_max(tf.abs(gradients[0]))

          samples_ = wf.sample(numsamples=numsamples,inputdim=2)
          samples = np.ones((numsamples, N), dtype=np.int32)

          inputs=tf.placeholder(dtype=tf.int32,shape=(None,N))
          log_amps=wf.log_amplitude(inputs,inputdim=2)

          local_energies = np.zeros(numsamples, dtype = np.complex64) #The type complex should be specified, otherwise the imaginary part will be discarded

          sigmas=np.zeros((2*N*numsamples,N), dtype=np.int32) #Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
          H = np.zeros(2*N*numsamples, dtype=np.float32) #Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
          log_amplitudes = np.zeros(2*N*numsamples, dtype=np.complex64) #Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)

          sigmaH = np.zeros((2*N,N), dtype = np.int32) #Array to store all the diagonal and non diagonal sigmas for each sample sigma
          matrixelements=np.zeros(2*N, dtype = np.float32) #Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

          for it in range(len(meanEnergy),numsteps+1):

              print("sampling started")

              start = time.time()

              samples=sess.run(samples_)

              end = time.time()
              print("sampling ended: "+ str(end - start))

              print("Magnetization = ", np.mean(2*samples - 1))

              #Getting the sigmas with the matrix elements
              slices, len_sigmas = J1J2Slices(J1,J2,Bz,samples, sigmas, H, sigmaH, matrixelements)

              #Process in steps to get log amplitudes
              # print("Generating log amplitudes Started")
              start = time.time()
              steps = len_sigmas//30000+1

              # print("number of required steps :" + str(steps))

              for i in range(steps):
                if i < steps-1:
                    cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
                else:
                    cut = slice((i*len_sigmas)//steps,len_sigmas)

                log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})
                # print(i+1)

              end = time.time()
              # print("Generating log amplitudes ended "+ str(end - start))

              #Generating the local energies
              for n in range(len(slices)):
                s=slices[n]
                local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

              meanE = np.mean(local_energies)
              varE = np.var(np.real(local_energies))

              #adding elements to be saved
              meanEnergy.append(meanE)
              varEnergy.append(varE)

              if it%1==0:
                  print('mean(E): {0} \pm {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it))

              if it%50==0:
                 np.save('../Check_Points/J1J2/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy',meanEnergy)
                 np.save('../Check_Points/J1J2/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy',varEnergy)

              if it>=5000 and varE <= np.min(varEnergy):
                 #Saving the performances if the model is better
                 saver.save(sess,path+'/'+filename)
                
              lr_ = 1/((1/lr)+(it/10)) #learning rate decay
              sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr_})
    #----------------------------------------
