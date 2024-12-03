#!/bin/bash -l

#PBS -N inversion
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#python3 -i -m pme.inversion --config config/clear.yaml

#python3 -i -m pme.inversion --config config/psf.yaml --noise 0.0
#python3 -i -m pme.inversion --config config/psf.yaml --noise 1.0e-2
#python3 -i -m pme.inversion --config config/psf.yaml --noise 1.0e-3
#python3 -i -m pme.inversion --config config/psf.yaml --noise 1.0e-4

#python3 -i -m pme.inversion --config config/no_psf.yaml --noise 0.0
#python3 -i -m pme.inversion --config config/no_psf.yaml --noise 1.0e-2
#python3 -i -m pme.inversion --config config/no_psf.yaml --noise 1.0e-3
#python3 -i -m pme.inversion --config config/no_psf.yaml --noise 1.0e-4


#python3 -m pme.inversion --config config/hinode.yaml
#python3 -m pme.inversion --config config/hinode_psf.yaml


#python3 -m pme.inversion --config config/muram_psf.yaml

#
python3 -m pme.inversion --config config/muram_sunspot.yaml
#python3 -m pme.inversion --config config/muram_sunspot_psf.yaml
