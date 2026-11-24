#!/bin/bash -l
#PBS -N prom3theus-hinode-mapper
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=12:00:00

PYTHONPATH=src python -m prom3theus.surrogate --config configs/muram/hinode/surrogate.yaml
