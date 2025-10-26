#!/bin/bash -l

#PBS -N global
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=08:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

# 90s cadence
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T00:00:00 --t_end 2024-05-09T06:00:00 --series 'S_90s' --cadence '90s'
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T06:00:00 --t_end 2024-05-09T12:00:00 --series 'S_90s' --cadence '90s'
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T12:00:00 --t_end 2024-05-09T18:00:00 --series 'S_90s' --cadence '90s'
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T18:00:00 --t_end 2024-05-10T00:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T00:00:00 --t_end 2024-05-10T06:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T06:00:00 --t_end 2024-05-10T12:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T12:00:00 --t_end 2024-05-10T18:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T18:00:00 --t_end 2024-05-11T00:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-11T00:00:00 --t_end 2024-05-11T06:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-11T06:00:00 --t_end 2024-05-11T12:00:00 --series 'S_90s' --cadence '90s'
