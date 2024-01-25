#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:15:00 2024

Generate a PSF

@author: mmolnar
"""

import scipy as sp 
import numpy as np 
import matplotlib.pyplot as pl


def gaussian(x, mu, sigma):
    
    result = np.exp(-1 * (x-mu)**2 
                    / (2 * sigma**2)) / (sigma * np.sqrt(2*np.pi))
    
    return result


if __name__ == "__main__":
    
    mu = 0 
    sigma = 1.5
    
    nx_halves = np.linspace(1,5, num=4)
    for el in nx_halves:
        nx = int(el * 2 + 1)
        dx = np.zeros((nx, nx))
    
        PSF = dx + 0.0
    
        for xx in range(nx):
            for yy in range(nx):
                dx[xx, yy] = ((xx - nx //2)** 2
                              + (yy - nx //2)** 2)
                PSF = gaussian(dx, mu, sigma)
    
        PSF = PSF / np.sum(PSF)
    
        np.savez(f"PSF_{nx}_x_{nx}_sigma_{sigma}.npz", 
                 PSF=PSF)