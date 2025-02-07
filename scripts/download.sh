#!/bin/bash -l

#PBS -N global
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=02:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T00:00:00 --t_end 2024-05-08T06:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T06:00:00 --t_end 2024-05-08T12:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T12:00:00 --t_end 2024-05-08T18:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T18:00:00 --t_end 2024-05-09T00:00:00


# 12 h cadence
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_12h --email robert.jarolim@uni-graz.at --t_start 2024-05-01T00:00:00 --t_end 2024-06-01T00:00:00 --cadence 12h


# test data
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test --email "robert.jarolim@uni-graz.at" --t_start "2024-05-01T00:00:00"


python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2 --email "robert.jarolim@uni-graz.at" --t_start "2024-05-08T00:00:00"
