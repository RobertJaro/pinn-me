#!/bin/bash -l

#PBS -N download
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=06:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

# download solar cycle
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2011-01-01T00:00:00 --t_end 2012-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2012-01-01T00:00:00 --t_end 2013-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2013-01-01T00:00:00 --t_end 2014-01-01T00:00:00 --cadence 1d
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2014-01-01T00:00:00 --t_end 2015-01-01T00:00:00 --cadence 1d
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2015-01-01T00:00:00 --t_end 2016-01-01T00:00:00 --cadence 1d
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2016-01-01T00:00:00 --t_end 2017-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2017-01-01T00:00:00 --t_end 2018-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2018-01-01T00:00:00 --t_end 2019-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2019-01-01T00:00:00 --t_end 2020-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2020-01-01T00:00:00 --t_end 2021-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2021-01-01T00:00:00 --t_end 2022-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2022-01-01T00:00:00 --t_end 2023-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2023-01-01T00:00:00 --t_end 2024-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2024-01-01T00:00:00 --t_end 2025-01-01T00:00:00 --cadence 1d
#python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/solar_cycle --email robert.jarolim@uni-graz.at --t_start 2024-01-01T00:00:00 --t_end 2025-01-01T00:00:00 --cadence 1d
