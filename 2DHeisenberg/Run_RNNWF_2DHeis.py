import tensorflow as tf
print("Tensorflow version =",tf.__version__)
import numpy as np
import os
import time
import random
import argparse
from Helper_Functions import *

from MDComplexRNN import RNNwavefunction
from MDTensorizedGRUcell import MDRNNcell
# from MDTensorizedcell import MDRNNcell #A cell without gating mechanism

parser = argparse.ArgumentParser()
parser.add_argument('--L', type = int, default=6)
parser.add_argument('--numunits', type = int, default=100)
parser.add_argument('--lr', type = float, default=5e-4)
parser.add_argument('--J2', type = float, default=1.0) # J2 = 1 -> triangular and J2 = 0.0 -> square lattice
parser.add_argument('--lrthreshold', type = float, default=5e-4)
parser.add_argument('--lrdecaytime', type = float, default=100)
parser.add_argument('--lrdecaytime_conv', type = float, default=2000)
parser.add_argument('--mag_fixed', type = str2bool, default=True)
parser.add_argument('--Sz', type = int, default=0)
parser.add_argument('--seed', type = int, default=111)
parser.add_argument('--RNN_symmetry', type = str, default='c4vsym')
parser.add_argument('--spinparity_fixed', type = str2bool, default=False)
parser.add_argument('--spinparity_value', type = int, default=1)
parser.add_argument('--group_character', type = str, default= 'A1')
parser.add_argument('--checkpoint_dir', type = str, default= "/checkpoint/hibatallah")
parser.add_argument('--gradient_clip', type = str2bool, default=False)
parser.add_argument('--gradient_clipvalue', type = float, default=10.0)
parser.add_argument('--normalized_gradients', type = str2bool, default=False)
parser.add_argument('--dotraining', type = str2bool, default=True)
parser.add_argument('--T0', type = float, default= 0.0)
parser.add_argument('--Nwarmup', type = int, default=1000)
parser.add_argument('--NannealingID', type = int, default=1)
parser.add_argument('--Ntrain', type = int, default=5)
parser.add_argument('--Nconvergence', type = int, default=0)
parser.add_argument('--numsamples', type = int, default=100)
# #To fine tune the model
parser.add_argument('--lrthreshold_conv', type = float, default=1e-4)

args = parser.parse_args()

num_layers = 1 #So far only one layer is supported
num_units = args.numunits
numsamples = args.numsamples
lr=args.lr
lrdecaytime = args.lrdecaytime
lrdecaytime_conv = args.lrdecaytime_conv
lrthreshold = args.lrthreshold
lrthreshold_conv = args.lrthreshold_conv
T0 = args.T0
mag_fixed = args.mag_fixed
magnetization = 2*args.Sz
spinparity_fixed = args.spinparity_fixed
spinparity_value = args.spinparity_value
RNN_symmetry = args.RNN_symmetry
group_character = args.group_character
J1 = +1
J2 = args.J2
gradient_clip = args.gradient_clip
gradient_clipvalue = args.gradient_clipvalue
normalized_gradients = args.normalized_gradients
dotraining = args.dotraining
Nwarmup = args.Nwarmup
Nannealing = args.NannealingID
Ntrain = args.Ntrain
Nconvergence = args.Nconvergence
numsteps = Nwarmup + (Nannealing+1)*Ntrain + Nconvergence

print("L =", args.L)
print("Num_units =", num_units)
print("J2 =", J2)
print("Num_steps =", numsteps)
print("Learning rate", lr, "Lr decaytime", lrdecaytime, "Lr threshold", lrthreshold)
print("Magnetization is fixed =", mag_fixed)
print("Magnetization (if fixed) =", magnetization)
print("Spin Parity is fixed =", spinparity_fixed)
print("Spin Parity value (if fixed) =", spinparity_value)
print("RNN symmetry =", RNN_symmetry)
print("Group character =", group_character)
print("Learning rate =", lr)
print("Gradient_clip =", gradient_clip)
print("Normalized gradients = ", normalized_gradients)
print("do training =", dotraining)
print("Pre-annealing temperature", T0)
print("Nwarmup, Nannealing, Ntrain, Nconvergence=", Nwarmup, Nannealing, Ntrain, Nconvergence)
print("numsamples =", numsamples, ", numsteps_conv =", Nconvergence, ", lrthreshold_conv =", lrthreshold_conv)

units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

Nx=args.L #x dim
Ny=args.L #y dim

numsamples_estimation = 2*10**4 #number of samples used for the estimation of the energy
numbatches = 2*10**4 #Maximum allowed number of samples to fed to the RNN to avoid memory allocation issues

################ Configuring Filenames ###########################

if not os.path.exists('./Check_Points/'):
    os.mkdir('./Check_Points/')

if not os.path.exists('./Check_Points/Size_'+str(Nx)+'x'+ str(Ny)):
    os.mkdir('./Check_Points/Size_'+str(Nx)+'x'+ str(Ny))

ending='_units'
for u in units:
    ending+='_{0}'.format(u)


savename = '_magfixed'+str(mag_fixed)+'_mag'+str(magnetization)

tf.compat.v1.reset_default_graph()
seed = args.seed
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

#Path for checkpoints
backgroundpath = './Check_Points/Size_'+str(Nx)+'x'+ str(Ny)
filename_checkpoint = args.checkpoint_dir+'/RNNwavefunction_2DTCRNN_'+str(Nx)+'x'+ str(Ny)+ending+savename+'.ckpt'

#If desired to checkpoint from a previous system size
backgroundpath_old = './Check_Points/Size_'+str(Nx-2)+'x'+ str(Ny-2)
filename_oldsyssize = backgroundpath_old+'/RNNwavefunction_2DTCRNN_'+str(Nx-2)+'x'+ str(Ny-2)+ending+savename+'.ckpt'

###############Intitializing the RNN---------------------------------------------------------------------------------

wf=RNNwavefunction(Nx,Ny,units=units,cell=MDRNNcell,seed = seed, mag_fixed = mag_fixed, magnetization = magnetization) #contains the graph with the RNNs

###############Running the training -------------------------------------------------------------------------------

meanEnergy=[]
varEnergy=[]

print('Training with numsamples = ', numsamples)
print('\n')

with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.compat.v1.placeholder(dtype=tf.float64,shape=[])

        samples_tensor = wf.sample(numsamples=numsamples,inputdim=2)
        inputs=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,Nx, Ny))

        Eloc=tf.compat.v1.placeholder(dtype=tf.complex128,shape=[numsamples])
        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,Nx,Ny])
        Temperature_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

        if RNN_symmetry == "c4vsym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_c4vsym_spinparity(inputs,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_c4vsym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_c4vsym(inputs,inputdim=2, group_character = group_character)
                log_amps_tensor=wf.log_amplitude_c4vsym(samples_placeholder,inputdim=2, group_character = group_character)
        elif RNN_symmetry == "c2vsym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_c2vsym_spinparity(inputs,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_c2vsym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_c2vsym(inputs,inputdim=2,group_character = group_character)
                log_amps_tensor=wf.log_amplitude_c2vsym(samples_placeholder,inputdim=2,group_character = group_character)
        elif RNN_symmetry == "c2dsym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_c2dsym_spinparity(inputs,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_c2dsym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_c2dsym(inputs,inputdim=2,group_character = group_character)
                log_amps_tensor=wf.log_amplitude_c2dsym(samples_placeholder,inputdim=2,group_character = group_character)
        elif RNN_symmetry == "c2sym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_c2sym_spinparity(inputs,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_c2sym_spinparity(samples_placeholder,inputdim=2, group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_c2sym(inputs,inputdim=2,group_character = group_character)
                log_amps_tensor=wf.log_amplitude_c2sym(samples_placeholder,inputdim=2,group_character = group_character)
        elif RNN_symmetry == "rotreducedsym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_rotreducedsym_spinparity(inputs,inputdim=2,group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_rotreducedsym_spinparity(samples_placeholder,inputdim=2,group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_rotreducedsym(inputs,inputdim=2,group_character = group_character)
                log_amps_tensor=wf.log_amplitude_rotreducedsym(samples_placeholder,inputdim=2,group_character = group_character)
        elif RNN_symmetry == "rotsym":
            if spinparity_fixed:
                log_amps=wf.log_amplitude_rotsym_spinparity(inputs,inputdim=2,group_character = group_character, spinparity_value = spinparity_value)
                log_amps_tensor=wf.log_amplitude_rotsym_spinparity(samples_placeholder,inputdim=2,group_character = group_character, spinparity_value = spinparity_value)
            else:
                log_amps=wf.log_amplitude_rotsym(inputs,inputdim=2,group_character = group_character)
                log_amps_tensor=wf.log_amplitude_rotsym(samples_placeholder,inputdim=2,group_character = group_character)
        elif RNN_symmetry == "nosym":
            log_amps=wf.log_amplitude_nosym(inputs,inputdim=2)
            log_amps_tensor=wf.log_amplitude_nosym(samples_placeholder,inputdim=2)

        cost = 2*tf.math.real(tf.reduce_mean(input_tensor=tf.multiply(tf.math.conj(log_amps_tensor),tf.stop_gradient(Eloc))) - tf.reduce_mean(input_tensor=tf.math.conj(log_amps_tensor))*tf.reduce_mean(input_tensor=tf.stop_gradient(Eloc))) + 4*Temperature_placeholder*( tf.reduce_mean(tf.real(log_amps_tensor)*tf.stop_gradient(tf.real(log_amps_tensor))) - tf.reduce_mean(tf.real(log_amps_tensor))*tf.reduce_mean(tf.stop_gradient(tf.real(log_amps_tensor))) )

        if gradient_clip:
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            clipped_gradients = [tf.clip_by_value(g, -gradient_clipvalue, gradient_clipvalue) for g in gradients]
            optstep=optimizer.apply_gradients(zip(clipped_gradients,variables),global_step=global_step)
        elif normalized_gradients:
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate_placeholder)
            gradients, variables = zip(*optimizer.compute_gradients(cost))
            gradients_normalized = []
            for i in range(len(gradients)):
                random_numbers = tf.random_uniform(dtype=tf.float64,shape=tf.shape(gradients[i]))
                gradients_normalized.append(tf.sign(gradients[i])*random_numbers)
            optstep=optimizer.apply_gradients(zip(gradients_normalized,variables),global_step=global_step)
        else:
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
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

## Counting the number of parameters
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
            saver.restore(sess,filename_checkpoint)
            print("Loading was successful!")
        except:
            try:
                print("Loading the model from old system size")
                saver.restore(sess,filename_oldsyssize)
                print("Loading was successful!")
            except:
                print("Loading was failed!")

        try:
            print("Trying to load energies!")

            meanEnergy=np.loadtxt(backgroundpath + '/meanEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt', converters={0: lambda s: complex(s.decode().replace('+-', '-'))}, dtype = complex).tolist()
            varEnergy=np.loadtxt(backgroundpath + '/varEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt').tolist()
        except:
            print("Failed! No need to load energies if running for the first time!")

# #-----------


with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with wf.graph.as_default():

        N = Nx*Ny
        samples = np.ones((numsamples, Nx, Ny), dtype=np.int32)

        sigmas=np.zeros(((4*N+1)*numsamples,Nx, Ny), dtype=np.int32)
        H = np.zeros((4*N+1)*numsamples, dtype=np.float64)
        log_amplitudes = np.zeros((4*N+1)*numsamples, dtype=np.complex128)

        sigmaH = np.zeros(((4*N+1), Nx, Ny), dtype = np.int32)
        matrixelements=np.zeros((4*N+1), dtype = np.float64)

        if dotraining: #If we want to train

            starting_step = len(meanEnergy)

            for it in range(starting_step,numsteps):

                start = time.time()

                if it+1<=Nwarmup+Nannealing*Ntrain: #Before the end of annealing
                    lr_adap = max(lrthreshold, lr/(1+it/lrdecaytime))
                elif it+1>Nwarmup+Nannealing*Ntrain: #After annealing -> finetuning the model during convergence
                    lr_adap = lrthreshold_conv/(1+(it-(Nwarmup+Nannealing*Ntrain))/lrdecaytime_conv)

                samples, local_energies = Get_Samples_and_Elocs(J2, Nx, Ny, samples, sigmas, H, log_amplitudes, sigmaH, matrixelements, samples_tensor, inputs, log_amps, sess, RNN_symmetry)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                meanEnergy.append(meanE)
                varEnergy.append(varE)

                ########################### Thermal pre-annealing #############################
                if T0 != 0:
                    if it+1<=Nwarmup:
                        if (it+1)%5==0:
                            print("Pre-annealing, warmup phase:", (it+1), "/", Nwarmup)
                        T = T0
                    elif it+1 > Nwarmup and it+1<=Nwarmup+Nannealing*Ntrain:
                        if (it+1)%5==0:
                            print("Pre-annealing, annealing phase:", (it+1-Nwarmup)//Ntrain, "/", Nannealing)
                        T = T0*(1-((it+1-Nwarmup)//Ntrain)/Nannealing)
                    else:
                        T = 0.0

                    if (it+1)%5 == 0:
                        print("Temperature = ", T)
                    ####### End of thermal pre-annealing -----------

                    ##### Computing Free energy ###########
                    log_amplitudes_ = sess.run(log_amps, feed_dict={inputs:samples})
                    meanF = np.mean(local_energies + 2*T*np.real(log_amplitudes_))
                    varF = np.var(local_energies + 2*T*np.real(log_amplitudes_))
                    ######################################
                else:
                    T = 0.0
                ##############################################################################

                if (it+1)%5==0 or it==0:
                    print("learning_rate =", lr_adap)
                    print("Magnetization =", np.mean(np.sum(2*samples-1, axis = (1,2))))
                    if T0 != 0:
                        print('mean(E): {0}, varE: {1}, meanF: {2}, varF: {3}, #samples {4}, #Step {5} \n\n'.format(meanE,varE,meanF, varF,numsamples, it+1))
                    elif T0 == 0.0:
                        print('mean(E): {0}, varE: {1}, #samples {2}, #Step {3} \n\n'.format(meanE,varE,numsamples, it+1))


                if (it+1)%100==0:
                    #Saving the performances for loading
                    saver.save(sess,filename_checkpoint)
                    np.savetxt(backgroundpath + '/meanEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt', meanEnergy)
                    np.savetxt(backgroundpath + '/varEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) +ending + savename +'.txt', varEnergy)

                #Optimize
                sess.run(optstep,feed_dict={Eloc:local_energies,samples_placeholder:samples,learningrate_placeholder: lr_adap, Temperature_placeholder:T})

                if (it+1)%5 == 0:
                    print("iteration ended during", time.time()-start)

            #End training loop

        ### Energy estimation at the end of annealing or end of convergence
        print("Saving the model parameters:")
        saver.save(sess,filename_checkpoint)

        print("Energy estimation phase started for", numsamples_estimation, "samples:")

        samples2_tensor = wf.sample(numsamples=numbatches,inputdim=2)
        samples2 = np.ones((numsamples_estimation, Nx, Ny), dtype=np.int32)

        local_energies2 = np.zeros(numsamples_estimation, dtype = np.complex128) #The type complex should be specified, otherwise the imaginary part will be discarded
        sigmas2=np.zeros(((4*N+1)*numbatches, Nx, Ny), dtype=np.int32)
        H2 = np.zeros((4*N+1)*numbatches, dtype=np.float64)
        log_amplitudes2 = np.zeros((4*N+1)*numbatches, dtype=np.complex128)

        print("sampling Started")
        start_sampling = time.time()

        for i in range(numsamples_estimation//numbatches):
            samples2[i*numbatches:(i+1)*numbatches]=sess.run(samples2_tensor)
            print("Sampling progress =", (i+1)/(numsamples_estimation//numbatches))
            print("Getting local energies started:")
            local_energies2[i*numbatches:(i+1)*numbatches] = Get_Elocs(J2, Nx, Ny, samples2[i*numbatches:(i+1)*numbatches], sigmas2, H2, log_amplitudes2, sigmaH, matrixelements, inputs, log_amps, sess, RNN_symmetry)
            print("Local energies computation ended!")

            print("sampling ended: "+ str(time.time() - start_sampling))

        meanE_final = np.mean(local_energies2)/N
        errE_final = np.sqrt(np.var(local_energies2)/numsamples_estimation)/N


        np.savetxt(backgroundpath + '/Energies_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) + '_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'_Nsteps'+str(numsteps)+'.txt', local_energies2)
        np.savetxt(backgroundpath + '/FinalEnergy_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) + '_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'_Nsteps'+str(numsteps)+'.txt', data_to_store)

########################## Plotting some useful figures ##############################
import matplotlib.pyplot as plt

savename += '_Nsteps'+str(numsteps)

fig = plt.figure()
plt.semilogy(varEnergy)
plt.xlabel("# Iterations")
plt.ylabel("VarE")
fig.savefig(backgroundpath + '/varEnergyPlot_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) + '_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'.png')

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

fig = plt.figure()
plt.semilogy(movingaverage(varEnergy, 50))
plt.xlabel("# Iterations")
plt.ylabel("VarE")
fig.savefig(backgroundpath + '/varEnergyPlot_2DTGCRNN_'+str(Nx)+'x'+ str(Ny) + '_lradap'+str(lr)+'_samp'+str(numsamples)+ending + savename +'_movavg50.png')
