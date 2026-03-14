#!/bin/bash -l
# ===== PBS =====
#PBS -N pinn-me
#PBS -A P22100000
#PBS -q main
#PBS -l job_priority=economy
#PBS -l select=4:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /glade/derecho/scratch/rjarolim/spinn_me/out.log

set -euo pipefail

# ===== Env =====
module load conda/latest
module load cuda/12.3.2
module load openmpi || true          # let Lmod handle swaps if needed
conda activate lightning

cd /glade/u/home/rjarolim/projects/PINN-ME

# ===== Nodes / rendezvous =====
sort -u "$PBS_NODEFILE" > nodes.txt
NNODES=$(wc -l < nodes.txt)
MASTER_ADDR=$(head -n1 nodes.txt)
MASTER_PORT=29500
NPROC_PER_NODE=4

# Runtime tuning
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
# On Derecho (Slingshot), this often helps stability/perf:
export NCCL_SOCKET_IFNAME=hsn0
# (If you hit hangs, also try: export NCCL_IB_DISABLE=1)

echo "Nodes: $(tr '\n' ' ' < nodes.txt)"
echo "MASTER_ADDR: $MASTER_ADDR  MASTER_PORT: $MASTER_PORT"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  OMP_NUM_THREADS=$OMP_NUM_THREADS"


# ===== Launch training =====
mpiexec -np "$NNODES" --map-by ppr:1:node --bind-to none \
  /bin/bash -lc '
    cd "$PBS_O_WORKDIR"
    RANK=${OMPI_COMM_WORLD_RANK}
    echo "node $(hostname -s): node_rank=${RANK}"
    module load conda/latest
    module load cuda/12.3.2
    module load openmpi || true          # let Lmod handle swaps if needed
    conda activate lightning
    cd /glade/u/home/rjarolim/projects/PINN-ME
    exec torchrun \
      --nnodes='"$NNODES"' \
      --nproc_per_node='"$NPROC_PER_NODE"' \
      --node_rank=${RANK} \
      --master_addr='"$MASTER_ADDR"' \
      --master_port='"$MASTER_PORT"' \
      -m pme.inversion_spherical \
        --config config/hmi/hmi_202405_90s_no_physics.yaml
  '

