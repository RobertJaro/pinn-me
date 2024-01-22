"""
Test the ME module
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from pme.train.me_equations import MEAtmosphere

def plot_all_Stokes(atmos):
    
    fig, ax = pl.subplots(2, 2, dpi=150)
    ax[0, 0].plot(atmos.lambdaGrid / 1e-10, atmos.I)
    ax[0, 0].set_title("Stokes I")
    
    ax[0, 1].plot(atmos.lambdaGrid / 1e-10, atmos.Q / atmos.I)
    ax[0, 1].set_title("Stokes Q/I")

    ax[1, 0].plot(atmos.lambdaGrid / 1e-10, atmos.U / atmos.I)
    ax[1, 0].set_title("Stokes U/I")
    
    ax[1, 1].plot(atmos.lambdaGrid / 1e-10, atmos.V / atmos.I)
    ax[1, 1].set_title("Stokes V/I")
    
    for el in ax.flatten():
        el.grid(alpha=0.1)
        
    ax[0, 0].set_ylabel("Intensity")   
    ax[1, 0].set_ylabel("Intensity")        
    ax[1, 0].set_xlabel("$\\lambda$ [$\\AA$]")        
    ax[1, 1].set_xlabel("$\\lambda$ [$\\AA$]")        

    
    pl.tight_layout() 
    pl.show()
    
if __name__ == '__main__':
    lambda0 = 6301.5080
    jUp = 2.0
    jLow = 2.0
    gUp = 1.5
    gLow = 1.83
    lambdaStart = 6300.8
    lambdaStep = 0.03
    nLambda = 50
    BField = 100.0
    theta = 20.0
    chi = 20.0
    vmac = 2.0
    damping = 0.2
    B0 = 0.8
    B1 = 0.2
    mu = 1.0
    vdop = 0.0
    kl = 5.0

    atmos = MEAtmosphere(lambda0, jUp, jLow, gUp, gLow,
                         lambdaStart, lambdaStep, nLambda, BField, theta, chi,
                         vmac, damping, B0, B1, mu,
                         vdop, kl)
    atmos.compute_all_Stokes()
    
    plot_all_Stokes(atmos)
    
