#!/bin/bash -l

#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

#################################################################################
# create data set
#python3 -m pme.data.create_test_set --out_path "/glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/data"

#################################################################################
# train

# no psf; no noise
#python3 -m pme.inversion --config config/test_set/clear.yaml
#python3 -m pme.inversion --config config/test_set/static_clear.yaml

# no psf; varying noise
#python3 -m pme.inversion --config config/test_set/no_psf.yaml --noise 0.0
#python3 -m pme.inversion --config config/test_set/no_psf.yaml --noise 1.0e-4
#python3 -m pme.inversion --config config/test_set/no_psf.yaml --noise 1.0e-3
#python3 -m pme.inversion --config config/test_set/no_psf.yaml --noise 1.0e-2

# psf; varying noise
#python3 -m pme.inversion --config config/test_set/psf.yaml --noise 0.0
#python3 -m pme.inversion --config config/test_set/psf.yaml --noise 1.0e-4
#python3 -m pme.inversion --config config/test_set/psf.yaml --noise 1.0e-3
#python3 -m pme.inversion --config config/test_set/psf.yaml --noise 1.0e-2

# psf; varying noise; static
#python3 -m pme.inversion --config config/test_set/static_psf.yaml --noise 0.0
#python3 -m pme.inversion --config config/test_set/static_psf.yaml --noise 1.0e-4
#python3 -m pme.inversion --config config/test_set/static_psf.yaml --noise 1.0e-3
#python3 -m pme.inversion --config config/test_set/static_psf.yaml --noise 1.0e-2

#################################################################################
# convert to npz files

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/clear_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/clear_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/static_clear_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/static_clear_v01.npz

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/no_psf_0.0_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/no_psf_0.0_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/no_psf_1.0e-4_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/no_psf_1.0e-4_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/no_psf_1.0e-3_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/no_psf_1.0e-3_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/no_psf_1.0e-2_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/no_psf_1.0e-2_v01.npz

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/psf_0.0_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/psf_0.0_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/psf_1.0e-4_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/psf_1.0e-4_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/psf_1.0e-3_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/psf_1.0e-3_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/psf_1.0e-2_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/psf_1.0e-2_v01.npz

python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/static_psf_0.0_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/static_psf_0.0_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/static_psf_1.0e-4_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/static_psf_1.0e-4_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/static_psf_1.0e-3_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/static_psf_1.0e-3_v01.npz
python3 -m pme.evaluation.pme_to_npz --input /glade/work/rjarolim/pinn_me/test_set/static_psf_1.0e-2_v01/inversion.pme --output /glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/static_psf_1.0e-2_v01.npz

#################################################################################
# plot performance as a function of noise

#python3 -i -m pme.evaluation.test_set.compare_noise --reference '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set/data/parameters_009.npz' --input '/glade/campaign/hao/radmhd/rjarolim/PINN-ME/test_set' --output '/glade/work/rjarolim/pinn_me/test_set/evaluation'
