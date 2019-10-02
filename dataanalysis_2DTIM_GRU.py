import numpy as np
import os
import time
import matplotlib.pyplot as plt

from numpy import convolve

Nx=12 #x dim
Ny=12 #y dim
numsamples = 1000
variances = []
rel_errors = []
lr = np.float64(1e-3)

units = [100]*3
savename = '_2DTIM'
ending='units'
for u in units:
    ending+='_{0}'.format(u)

for test in range(1,2):

    E1 = -2.4096*(12*12)
    E2 = -3.17388*(12*12)
    E3 = -4.12178*(12*12)
    Energies = [E1,E2,E3]

    def movingaverage (values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    #-------------------
    fig = plt.figure(figsize=(10,5))

    colors = ["r-","b-", "g-"]
    for h in [2,3,4]:
        # meanEnergy =  np.load('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lr'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy')
        meanEnergy =  np.load('../Check_Points/2DTIM/GRU/meanEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy')

        rel_error = np.abs((meanEnergy - Energies[h-2])/Energies[h-2])

        rel_error_mov = movingaverage(rel_error,100)
        plt.semilogy(np.arange(1, len(rel_error_mov)+1), rel_error_mov, colors[h-2], label = "$h=" + str(h) +"$")

    plt.xlabel("Training step", fontsize=20)
    plt.ylabel("Relative error", fontsize=20)

    plt.xticks(np.arange(0,len(rel_error),10000),fontsize=20)
    plt.yticks([1e-5,1e-4,1e-4,1e-3,1e-2,1e-1],fontsize=20)
    plt.legend(fontsize=20)

    plt.rcParams['axes.facecolor'] = 'white'
    plt.style.use('default')
    plt.grid(None)

    # fig.savefig('figs/relativeerror_GRURNN_2DTIM_12x12_samp'+str(numsamples)+'_units_'+str(units[0])+'_lr'+str(lr)+'_test'+str(test)+'.png', bbox_inches="tight", dpi = 300)
    fig.savefig('figs/relativeerror_GRURNN_2DTIM_12x12_samp'+str(numsamples)+'_units_'+str(units[0])+'_'+str(len(units))+'_lradap'+str(lr)+'_test'+str(test)+'.png', bbox_inches="tight", dpi = 300)

    #------------------

    #-------------------
    # fig = plt.figure(figsize=(10,5))
    #
    #
    # for h in [2,3,4]:
    #     # varEnergy = np.load('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lr'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy')
    #     varEnergy = np.load('../Check_Points/2DTIM/GRU/varEnergy_GRURNN_'+str(Nx)+'x'+ str(Ny) +'_h'+str(h)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'_test'+str(test)+'.npy')
    #
    #     varEnergy_mov = movingaverage(varEnergy,100)
    #     plt.semilogy(np.arange(1, len(varEnergy_mov)+1), varEnergy_mov, label = "$h=" + str(h) +"$")
    #
    # plt.xlabel("Training step", fontsize=20)
    # plt.ylabel("Energy Variance", fontsize=20)
    #
    # plt.xticks(np.arange(0,len(varEnergy),5000),fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=20)
    #
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.style.use('default')
    # plt.grid(None)
    #
    # # fig.savefig('figs/energyvariance_GRURNN_2DTIM_12x12_samp'+str(numsamples)+'_units_'+str(units[0])+'_lr'+str(lr)+'_test'+str(test)+'.png', bbox_inches="tight", dpi = 300)
    # fig.savefig('figs/energyvariance_GRURNN_2DTIM_12x12_samp'+str(numsamples)+'_units_'+str(units[0])+'_lradap'+str(lr)+'_test'+str(test)+'.png', bbox_inches="tight", dpi = 300)

    #------------------
