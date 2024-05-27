import tensorflow as tf
print("Tensorflow version =",tf.__version__)
import numpy as np
import os
import time
import random
import argparse
from Helping_Functions import *
import matplotlib.pyplot as plt

from MDRNNWavefunction import MDRNNWavefunction
from MDGRUcell import MDRNNcell

parser = argparse.ArgumentParser()
parser.add_argument('--Lx', type = int, default=2)
parser.add_argument('--Ly', type = int, default=2)
parser.add_argument('--numunits', type = int, default=40)
parser.add_argument('--lr', type = float, default=1e-3)
parser.add_argument('--lrthreshold', type = float, default=1e-4)
parser.add_argument('--seed', type = int, default=111)
parser.add_argument('--gradient_clip', type = str2bool, default=False)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--normalized_gradients', type = str2bool, default=False)
parser.add_argument('--dotraining', type = str2bool, default=True)
parser.add_argument('--numsamples', type = int, default=500)
parser.add_argument('--L_cut', type = int, default=0)
parser.add_argument('--T0', type = float, default=1.0)
parser.add_argument('--Rcutoff', type = float, default=2.001) #+0.001 is to avoid numerical float precision issues
parser.add_argument('--delta', type = float, default=3.3)
parser.add_argument('--Nannealing', type = int, default=10000)

## To fine tune the model
parser.add_argument('--Nconvergence', type = int, default=10000)

# Hamiltonian parameters
parser.add_argument('--Rb', type = float, default=1.7)
parser.add_argument('--Omega', type = float, default=1.0)
parser.add_argument('--deltaID', type = int, default=0)

args = parser.parse_args()

seed = args.seed
num_layers = 1
num_units = args.numunits
numsamples = args.numsamples
numsamples = args.numsamples
lr=args.lr
lrthreshold = args.lrthreshold
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
normalized_gradients = args.normalized_gradients
dotraining = args.dotraining
T0 = args.T0
# Hamiltonian parameters
Rb = round(args.Rb,2)     # Strength of Van der Waals interaction
a = 1.0
Omega = args.Omega # Rabi frequency
V = Omega*(Rb**6)/(a**6) # constant in the van der waals potential V = C/R^6
delta = round(args.delta,2) # Detuning
Rcutoff = args.Rcutoff

print("Lx =", args.Lx, "Ly =", args.Ly)
print("Num_units =", num_units)
print("Learning rate for warmup", lr, "Lr after warmup", lrthreshold)
print("Rb =", Rb, ", Omega =", Omega, ", delta =", delta)
print("Learning rate =", lr)
print("Gradient_clip =", gradient_clip)
print("Normalized gradients = ", normalized_gradients)
print("do training =", dotraining)


#### Annealing parameters
numwarmup = 1000
Nannealing = args.Nannealing
Neq = 5
numsteps = numwarmup + Nannealing*Neq + args.Nconvergence
print("Warmup steps", numwarmup, "Nanneal",  Nannealing, "Neq", Neq, "Nconv", args.Nconvergence)

units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

Nx=args.Lx #x dim
Ny=args.Ly #y dim

numsamples_estimation_entropy = 10**7
numsamples_estimation = 2*10**5
numsamples_correlations = numsamples_estimation
numbatches = 2*10**4
numsamples_xcorrelations = 2*10**4
# ################ Configuring Filenames ###########################

ending='_units'
for u in units:
    ending+='_{0}'.format(u)

savename = '_2DGRNN_samp'+str(numsamples)+ending+'_lrthres'+str(lrthreshold)+"_seed"+str(seed)+"_Nanneal"+str(Nannealing)

if gradient_clip:
    savename += '_clippedgradient'+str(gradient_clipvalue)

savename_nosym = savename

savename += '_Rb'+str(Rb)+"_delta"+str(delta)
savename_nosym += '_Rb'+str(Rb)+"_delta"+str(delta)

# #####################################################################################


tf.compat.v1.reset_default_graph()
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

if not os.path.exists('./Check_Points_Rcutoff'+str(Rcutoff)):
    os.mkdir('./Check_Points_Rcutoff'+str(Rcutoff))

if not os.path.exists('./Check_Points_Rcutoff'+str(Rcutoff)+'/Annealing'):
    os.mkdir('./Check_Points_Rcutoff'+str(Rcutoff)+'/Annealing')

if not os.path.exists('./Check_Points_Rcutoff'+str(Rcutoff)+'/Annealing/Size_'+str(Nx)+'x'+ str(Ny)):
    os.mkdir('./Check_Points_Rcutoff'+str(Rcutoff)+'/Annealing/Size_'+str(Nx)+'x'+ str(Ny))

# #Path for checkpoints
backgroundpath = './Check_Points_Rcutoff'+str(Rcutoff)+'/Annealing/Size_'+str(Nx)+'x'+ str(Ny)
filename = backgroundpath+'/RNNwavefunction'+savename+'.ckpt'

###############Intitializing the RNN---------------------------------------------------------------------------------

wf=MDRNNWavefunction(Nx,Ny,units=units,cell=MDRNNcell,inputdim = 2, seed = seed) #contains the graph with the RNNs

###############Running the training -------------------------------------------------------------------------------

meanEnergy=[]
varEnergy=[]
meanFreeEnergy=[]
varFreeEnergy=[]

print('Training with numsamples = ', numsamples)
print('\n')

with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
        Temperature_placeholder=tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
        learning_rate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 200, 1.00, staircase=True) #decay every 10 step

        samples_tensor = wf.sample(numsamples=numsamples,inputdim=2)
        inputs_placeholder=tf.compat.v1.placeholder(dtype=tf.int64,shape=(None,Nx,Ny,3))

        Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int64,shape=[numsamples,Nx,Ny,3])

        log_probs_tensor=wf.log_probability_symmetry(inputs_placeholder,inputdim=2, symmetry = RNN_symmetry)
        log_probs_forgrad=wf.log_probability_symmetry(samples_placeholder,inputdim=2, symmetry = RNN_symmetry)

        Floc = Eloc + Temperature_placeholder*log_probs_forgrad
        cost = tf.reduce_mean(input_tensor=tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(input_tensor=log_probs_forgrad)*tf.reduce_mean(input_tensor=tf.stop_gradient(Floc))

        # cost = tf.reduce_mean(input_tensor=tf.multiply(log_probs_forgrad,tf.stop_gradient(Eloc))) - tf.reduce_mean(input_tensor=log_probs_forgrad)*tf.reduce_mean(input_tensor=tf.stop_gradient(Eloc))

        if gradient_clip:
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            clipped_gradients = [tf.clip_by_value(g, -gradient_clipvalue, gradient_clipvalue) for g in gradients]
            optstep=optimizer.apply_gradients(zip(clipped_gradients,variables),global_step=global_step)
        elif normalized_gradients:
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients_normalized = []
            for i in range(len(gradients)):
                random_numbers = tf.random_uniform(dtype=tf.float64,shape=tf.shape(gradients[i]))
                gradients_normalized.append(tf.sign(gradients[i])*random_numbers)
            optstep=optimizer.apply_gradients(zip(gradients_normalized,variables),global_step=global_step)
        else:
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)

        #End calculate Gradients---------------


        init=tf.compat.v1.global_variables_initializer()
        saver=tf.compat.v1.train.Saver() #define tf saverJ

#Starting Session------------
#Activating GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess=tf.compat.v1.Session(graph=wf.graph, config=config)
sess.run(init)
#------------------------------------------------------------------------------------------------------------------------------

## Cuunting the number of parameters
with wf.graph.as_default():
    variables_names =[v.name for v in tf.compat.v1.trainable_variables()]
    sum = 0
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        v1 = tf.reshape(v,[-1])
        print(k,v1.shape)
        sum +=v1.shape[0]
    print('The sum of params is {0}'.format(sum))

# #Loading previous trainings-----------------------------------------------------------------------------------------------------
with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        try:
            print("Loading the model from checkpoint")
            saver.restore(sess,filename)
            print("Loading was successful!")
        except:
            print("Loading was failed!")

        try:
            print("Trying to load energies!")

            meanEnergy=np.loadtxt(backgroundpath + '/meanEnergy'+ savename +'.txt').tolist()
            varEnergy=np.loadtxt(backgroundpath + '/varEnergy' + savename +'.txt').tolist()
            meanFreeEnergy=np.loadtxt(backgroundpath + '/meanFreeEnergy'+ savename +'.txt').tolist()
            varFreeEnergy=np.loadtxt(backgroundpath + '/varFreeEnergy' + savename +'.txt').tolist()
        except:
            print("Failed! No need to load energies if running for the first time!")


with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        N = Nx*Ny*3
        samples = np.ones((numsamples, Nx, Ny), dtype=np.int32)

        queue_samples = np.zeros((Nx*Ny*3+1, numsamples, Nx,Ny, 3), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
        log_probs_memory = np.zeros((Nx*Ny*3+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

        kagome_lattice = generate_kagome(Nx, Ny)
        neighbours, distances = neighbouring_atoms(kagome_lattice, Rcutoff, Nx, Ny)

        if dotraining:

            T = T0

            for it in range(len(meanEnergy)+1,numsteps+1):

                start = time.time()

                if it>=numwarmup + Nannealing*Neq: #If finetuning the model during convergence
                    lr_adap = lrthreshold
                else:
                    lr_adap = lr

                #Annealing
                if it>=numwarmup and it < numwarmup + Nannealing*Neq:
                    if it%Neq == 0:
                        T = T0*(1-(it-numwarmup)/(Nannealing*Neq))
                        print("Nanneal step", (it-numwarmup)/Neq, "/", Nannealing)
                elif it > numwarmup + Nannealing*Neq:
                    T = 0.0

                samples = sess.run(samples_tensor)
                log_probabilities= sess.run(log_probs_tensor, feed_dict={inputs_placeholder:samples})
                local_energies = Rydberg_local_energies(V, delta, Omega, neighbours, distances, samples, queue_samples, log_probs_tensor, inputs_placeholder, log_probs_memory, sess, RNN_symmetry)

                meanE = np.mean(local_energies)/N
                varE = np.var(local_energies)/N

                meanF = np.mean(local_energies + T*log_probabilities)/N
                varF = np.var(local_energies + T*log_probabilities)/N

                meanEnergy.append(meanE)
                varEnergy.append(varE)
                meanFreeEnergy.append(meanF)
                varFreeEnergy.append(varF)

                if it%5==0:
                    print("learning_rate =", lr_adap)
                    print("Temperature =", T)
                    print('mean(E): {0}, varE: {1}, mean(F): {2}, var(F): {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE,meanF,varF,numsamples, it))

                if it%200==0:
                    #Saving the performances for loading
                    saver.save(sess,filename)
                    np.savetxt(backgroundpath + '/meanEnergy' + savename +'.txt', meanEnergy)
                    np.savetxt(backgroundpath + '/varEnergy' + savename +'.txt', varEnergy)
                    np.savetxt(backgroundpath + '/meanFreeEnergy' + savename +'.txt', meanFreeEnergy)
                    np.savetxt(backgroundpath + '/varFreeEnergy' + savename +'.txt', varFreeEnergy)

                #Optimize
                sess.run(optstep,feed_dict={Eloc:local_energies,samples_placeholder:samples,learningrate_placeholder: lr_adap, Temperature_placeholder: T})

                if it%5 == 0:
                    print("iteration ended during", time.time()-start)

            #End training loop

        print("Saving the model parameters:")
        saver.save(sess,filename)

        print("Energy estimation phase started for", numsamples_estimation, "samples:")

        samples2_tensor = wf.sample(numsamples=numbatches,inputdim=2)
        samples2 = np.ones((numsamples_estimation, Nx, Ny, 3), dtype=np.int32)

        queue_samples2 = np.zeros((Nx*Ny*3+1, numbatches, Nx,Ny,3), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
        log_probs_memory2 = np.zeros((Nx*Ny*3+1)*numbatches, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
        local_energies2 = np.zeros(numsamples_estimation, dtype=np.float64)

        print("sampling Started")
        start_sampling = time.time()

        for i in range(numsamples_estimation//numbatches):
            samples2[i*numbatches:(i+1)*numbatches]=sess.run(samples2_tensor)
            print("Sampling progress =", (i+1)/(numsamples_estimation//numbatches))
            print("Getting local energies started:")
            local_energies2[i*numbatches:(i+1)*numbatches] = Rydberg_local_energies(V, delta, Omega, neighbours, distances, samples2[i*numbatches:(i+1)*numbatches], queue_samples2, log_probs_tensor, inputs_placeholder, log_probs_memory2, sess, RNN_symmetry)
            print("Local energies computation ended!")

            print("sampling ended: "+ str(time.time() - start_sampling))

        meanE_final = np.mean(local_energies2)/N
        errE_final = np.sqrt(np.var(local_energies2)/numsamples_estimation)/N
        print("Energy per site:", meanE_final)
        print("Error on energy:", errE_final)

        L_cut = args.L_cut
        print("L_cut =", L_cut)

        ##################################################################################################################################

        print("Computing ground state densities")
        kagome_lattice = generate_kagome(Nx, Ny)

        densities = np.mean(samples2, axis = 0)
        densities_err = np.std(samples2, axis = 0)/np.sqrt(samples2.shape[0])

        avg_density = np.mean(samples2)
        avg_density_err = np.std(np.mean(samples2, axis = (1,2,3)))/np.sqrt(samples2.shape[0])
        print("Overall avg density =", avg_density, "err =", avg_density_err)

        densities_reshaped = np.array([densities[index//(3*Ny),(index%Ny)//(3),index%3] for index in range(Nx*Ny*3)])

        fig,ax = plt.subplots()

        plt.scatter(kagome_lattice[:,0], kagome_lattice[:,1], c = densities_reshaped, cmap = "rainbow", rasterized=True, vmin = 0, vmax = 1)
        plt.xlabel("$X$")
        plt.ylabel("$Y$")
        cbar = plt.colorbar()
        cbar.set_label(r'Density')
        cbar.ax.tick_params()

        fig.savefig(backgroundpath + '/Density'+ savename +'_Lcut'+str(L_cut)+'.png', dpi = 300)
