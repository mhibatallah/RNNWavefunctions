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

def inverse_one_hot(samples_onehot):
    return np.argmax(samples_onehot, axis=1)

def binary_to_intarray(binary_samples):
    "This function transforms a binary array of size (_,numsamples, Nx,Ny, 3) to (_,numsamples, Nx,Ny) where integers goes between 0 and 7"
    _,numsamples,Nx,Ny,bin_length = binary_samples.shape
    prod_window = np.array([2**i for i in range(bin_length)])
    return np.tensordot(binary_samples, prod_window, axes = [4,0])

def intarray_to_binary(samples, bin_length = 3):
    "This function does the inverse of binary_to_intarray"
    numsamples,Nx,Ny= samples.shape
    samples_new = np.array(samples, dtype=np.uint8).reshape(numsamples, Nx, Ny, 1)
    return np.unpackbits(samples_new, axis=3)[:,:,:,::-1][:,:,:,:bin_length]

#####################################################################
def generate_kagome(Nx, Ny):
    kagome_lattice = np.ones((Nx*Ny*3,2))
    for i in range(Nx):
        for j in range(Ny):
            #left spin
            kagome_lattice[i*Ny*3+j*3+1,0] = 2*(i + j/2)
            kagome_lattice[i*Ny*3+j*3+1,1] = 2*(np.sqrt(3)/2 * j)

            #right spin
            kagome_lattice[i*Ny*3+j*3+2,0] = kagome_lattice[i*Ny*3+j*3+1,0] + 1
            kagome_lattice[i*Ny*3+j*3+2,1] = kagome_lattice[i*Ny*3+j*3+1,1]

            #up spin
            kagome_lattice[i*Ny*3+j*3+0,0] = kagome_lattice[i*Ny*3+j*3+1,0] + 1/2
            kagome_lattice[i*Ny*3+j*3+0,1] = kagome_lattice[i*Ny*3+j*3+1,1] + np.sqrt(3)/2
    return kagome_lattice

def neighbouring_atoms(positions, cutoff, Nx, Ny):
    neighbours = [[], [], []]
    distances = [[], [], []]
    N = Nx*Ny*3
    #In each unit cell of 3 spins
    index_upspin = 0
    index_leftspin = 1
    index_rightspin = 2

    a1 = np.array([2,0])
    a2 = np.array([1,np.sqrt(3)])

    positions_xtranslated = np.copy(positions)-Nx*a1

    positions_ytranslated = np.copy(positions)-Ny*a2

    positions_xytranslated = np.copy(positions)-Nx*a1-Ny*a2

    for i in range(Nx):
        for j in range(Ny):
            for k in range(3):
                for k_cell in range(3):
                    distance = np.sqrt(np.sum((positions[i*Ny*3+j*3+k]-positions[k_cell])**2))
                    if distance > 0 and distance <= cutoff:
                        neighbours[k_cell].append([i,j,k])
                        distances[k_cell].append(distance)

                    #Taking PBC into account
                    distance = np.sqrt(np.sum((positions_xtranslated[i*Ny*3+j*3+k]-positions[k_cell])**2))
                    if distance > 0 and distance <= cutoff:
                        neighbours[k_cell].append([i,j,k])
                        distances[k_cell].append(distance)

                    #Taking PBC into account
                    distance = np.sqrt(np.sum((positions_ytranslated[i*Ny*3+j*3+k]-positions[k_cell])**2))
                    if distance > 0 and distance <= cutoff:
                        neighbours[k_cell].append([i,j,k])
                        distances[k_cell].append(distance)


                    #Taking PBC into account
                    distance = np.sqrt(np.sum((positions_xytranslated[i*Ny*3+j*3+k]-positions[k_cell])**2))
                    if distance > 0 and distance <= cutoff:
                        neighbours[k_cell].append([i,j,k])
                        distances[k_cell].append(distance)

    return neighbours, distances

###################################################################################

def Rydberg_local_energies(V, delta, Omega, neighbours, distances, binary_samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ Computing the local energies of the Rydberg Hamiltonian on a Kagome Lattice in 2D
    """

    numsamples,Nx,Ny, numspins_percell = binary_samples.shape

    N = numspins_percell*Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    local_energies += -np.sum(delta*binary_samples, axis=(1,2,3))

    #In each unit cell of 3 spins
    index_upspin = 0
    index_leftspin = 1
    index_rightspin = 2
    for j in range(Ny):
        for i in range(Nx):
            for k_cell in range(3):
                for (k,elt) in enumerate(neighbours[k_cell]):
                    local_energies += (V/2)*binary_samples[:,i,j,k_cell]*binary_samples[:,(i+elt[0])%Nx,(j+elt[1])%Ny,elt[2]]/(distances[k_cell][k]**6)

    queue_samples[0] = binary_samples #storing the diagonal samples

    #Non-diagonal elements---------------
    if Omega != 0:
        for j in range(Ny):
            for i in range(Nx):
                for k in range(numspins_percell):
                    valuesT = np.copy(binary_samples)
                    valuesT[:,i,j,k][binary_samples[:,i,j,k]==1] = 0 #Flip
                    valuesT[:,i,j,k][binary_samples[:,i,j,k]==0] = 1 #Flip

                    queue_samples[(j*Nx+i)*numspins_percell+k+1] = valuesT

    #Calculating log_probs from samples

    len_sigmas = (N+1)*numsamples
    steps = ceil(numsampsteps*len_sigmas/25000) #I want a maximum of 25000 in batch size just to not allocate too much memory
    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, Nx,Ny, numspins_percell])
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
    log_probs_diagonal = log_probs_reshaped[0,:]
    log_probs_nondiagonal = log_probs_reshaped[1:,:]

    local_energies += -(Omega/2)*np.sum(np.exp(0.5*log_probs_nondiagonal-0.5*log_probs_diagonal), axis = 0)

    return local_energies
