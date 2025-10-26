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

python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-07T00:00:00 --t_end 2024-05-07T12:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-07T12:00:00 --t_end 2024-05-08T00:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T00:00:00 --t_end 2024-05-08T12:00:00
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405 --email robert.jarolim@uni-graz.at --t_start 2024-05-08T12:00:00 --t_end 2024-05-09T00:00:00

# 12 h cadence
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_12h --email robert.jarolim@uni-graz.at --t_start 2024-05-01T00:00:00 --t_end 2024-06-01T00:00:00 --cadence 12h


# test data
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test --email "robert.jarolim@uni-graz.at" --t_start "2024-05-01T00:00:00"
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2 --email "robert.jarolim@uni-graz.at" --t_start "2024-05-08T00:00:00"
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2024_05_09 --email "robert.jarolim@uni-graz.at" --t_start "2024-05-09T01:00:00"
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2024_03_24 --email "robert.jarolim@uni-graz.at" --t_start "2024-03-24T01:00:00"
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2024_03_15 --email "robert.jarolim@uni-graz.at" --t_start "2024-03-15T06:00:00"
python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/test_2024_03_27 --email "robert.jarolim@uni-graz.at" --t_start "2024-03-27T06:00:00"


# 11158
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/201102_3h --email "robert.jarolim@uni-graz.at" --t_start "2011-02-15T15:00:00" --t_end "2011-02-20T00:00:00" --cadence "3h"

python3 -m pme.data.download_hmi_test --download_dir /glade/work/rjarolim/data/hmi_stokes/201102_test --email "robert.jarolim@uni-graz.at" --t_start "2011-02-15T00:00:00"



python download_hmi.py --download_dir "/home/memolnar/Data/PINN-ME/HMI/20240505" --email "momchil.molnar@gmail.com" --t_start "2024/05/05 12:00:00" --t_end "2024/05/05 13:00:00"

python download_hmi.py --download_dir "/home/memolnar/Data/PINN-ME/HMI/20151109_135s/" --email "momchil.molnar@gmail.com" --t_start "2015/11/09 11:00:00" --t_end "2015/11/09 12:00:00" --series 'S_135s'

# 90s cadence
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T00:00:00 --t_end 2024-05-09T06:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T06:00:00 --t_end 2024-05-09T12:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T12:00:00 --t_end 2024-05-09T18:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-09T18:00:00 --t_end 2024-05-10T00:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T00:00:00 --t_end 2024-05-10T06:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T06:00:00 --t_end 2024-05-10T12:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T12:00:00 --t_end 2024-05-10T18:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-10T18:00:00 --t_end 2024-05-11T00:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-11T00:00:00 --t_end 2024-05-11T06:00:00 --series 'S_90s' --cadence '90s'
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/202405_90s --email robert.jarolim@uni-graz.at --t_start 2024-05-11T06:00:00 --t_end 2024-05-11T12:00:00 --series 'S_90s' --cadence '90s'

# 90s cadence
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/20240410_90s --email robert.jarolim@uni-graz.at --t_start 2024-04-10T00:00:00 --t_end 2024-04-10T02:00:00 --series 'S_90s' --cadence '90s'


# 2024-03-23
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/20240323_720s --email robert.jarolim@uni-graz.at --t_start 2024-03-23T22:15:00 --t_end 2024-03-24T02:15:00 --series 'S_720s' --cadence '720s'

# 2024-03-15
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/20240315_720s --email robert.jarolim@uni-graz.at --t_start 2024-03-15T00:00:00 --t_end 2024-03-15T18:00:00 --series 'S_720s' --cadence '720s'

# 2024-03-27
python3 -m pme.data.download_hmi --download_dir /glade/work/rjarolim/data/hmi_stokes/20240327_720s --email robert.jarolim@uni-graz.at --t_start 2024-03-27T00:00:00 --t_end 2024-03-28T00:00:00 --series 'S_720s' --cadence '6h'
