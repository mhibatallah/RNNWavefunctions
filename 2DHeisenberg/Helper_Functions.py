import numpy as np
import time
from math import ceil

#######################################################################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
######################################################################


###################################################################################
def Triangular2DHeis_MatrixElements(J2,Nx,Ny,sigmap,sigmas, matrixelements):
    """
    J2 is the diagonal couplings between spins
    J2 = 1.0 corresponds to the isotropic triangular lattice
    J2 = 0.0 corresponds to the Heisenberg square lattice
    """
    #the diagonal part is simply the sum of all Sz-Sz interactions
    diag=0
    num = 0 #Number of basis elements

    for nx in range(Nx):
        for ny in range(Ny):
            if nx != (Nx-1):  #if not on the right
                if sigmap[nx,ny] != sigmap[(nx+1),ny]:
                    diag-=0.25 #add a negative energy contribution
                else:
                    diag+=0.25

            if ny != (Ny-1): #if not on the down
                if sigmap[nx,ny] != sigmap[nx,(ny+1)]:
                    diag-=0.25 #add a negative energy contribution
                else:
                    diag+=0.25

            if ny != (Ny-1) and nx != (Nx-1): #nnn
                if sigmap[nx,ny] != sigmap[(nx+1),(ny+1)]:
                    diag-=0.25*J2 #add a negative energy contribution
                else:
                    diag+=0.25*J2


    matrixelements[num] = diag #add the diagonal part to the matrix elements
    sig = np.copy(sigmap)

    sigmas[num] = sig
    num += 1

    #off-diagonal part (For the Heis Model)
    for ny in range(Ny):
        for nx in range(Nx):
            if nx != (Nx-1):  #if not on the right
                if sigmap[nx,ny] != sigmap[(nx+1),ny]:
                    sig = np.copy(sigmap)
                    sig_temp = sig[nx,ny]
                    sig[nx,ny] = sig[(nx+1),ny]
                    sig[(nx+1),ny] = sig_temp

                    sigmas[num] = sig
                    matrixelements[num] = -1/2 #negatice sign means Marshal sign is implemented
                    num += 1

            if ny != (Ny-1): #if not on the down
                if sigmap[nx,ny] != sigmap[nx,(ny+1)]:
                    sig = np.copy(sigmap)
                    sig_temp = sig[nx,ny]
                    sig[nx,ny] = sig[nx,(ny+1)]
                    sig[nx,(ny+1)] = sig_temp

                    sigmas[num] = sig
                    matrixelements[num] = -1/2
                    num += 1

            if ny != (Ny-1) and nx != (Nx-1): #nnn

                if sigmap[nx,ny] != sigmap[(nx+1),(ny+1)]:
                    sig = np.copy(sigmap)
                    sig_temp = sig[nx,ny]
                    sig[nx,ny] = sig[(nx+1),(ny+1)]
                    sig[(nx+1),(ny+1)] = sig_temp

                    sigmas[num] = sig
                    matrixelements[num] = +J2/2
                    num += 1

    return num

def Triangular2DHeis_LocalEnergies(J2,Nx,Ny,sigmasp,sigmas,H,sigmaH,matrixelements):
    """
    """
    slices=[]
    sigmas_length = 0

    numsamples =sigmasp.shape[0]

    for n in range(numsamples):
        sigmap=sigmasp[n,:]
        num = Triangular2DHeis_MatrixElements(J2,Nx,Ny,sigmap, sigmaH, matrixelements)
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        if (len(H[s])!=num):
            print("error")
            print(H[s].shape,s, matrixelements[:num].shape)

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas


    return slices,sigmas_length
########################################################################################


def Get_Samples_and_Elocs(J2,Nx, Ny, samples, sigmas, H, log_amplitudes, sigmaH, matrixelements, samples_tensor, inputs, log_amps, sess, RNN_symmetry):

    local_energies = np.zeros(samples.shape[0], dtype = np.complex128) #The type complex should be specified, otherwise the imaginary part will be discarded

    samples=sess.run(samples_tensor)

    slices, len_sigmas = Triangular2DHeis_LocalEnergies(J2,Nx,Ny,samples, sigmas, H, sigmaH, matrixelements)

    if RNN_symmetry == "c2dsym":
        numsampsteps = 4
    else:
        numsampsteps = 1

    steps = ceil(numsampsteps*len_sigmas/20000)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)

        log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})

    for n in range(len(slices)):
        s=slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

    return samples, local_energies

def Get_Elocs(J2, Nx, Ny, samples, sigmas, H, log_amplitudes, sigmaH, matrixelements, inputs, log_amps, sess, RNN_symmetry):

    local_energies = np.zeros(samples.shape[0], dtype = np.complex128) #The type complex should be specified, otherwise the imaginary part will be discarded

    slices, len_sigmas = Triangular2DHeis_LocalEnergies(J2,Nx,Ny,samples,sigmas, H, sigmaH, matrixelements)

    if RNN_symmetry == "c2dsym":
        numsampsteps = 4
    else:
        numsampsteps = 1
    steps = ceil(numsampsteps*len_sigmas/20000)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)

        log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})
        print("Step:", i+1, "/", steps)

    for n in range(len(slices)):
        s=slices[n]
        local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

    return local_energies
