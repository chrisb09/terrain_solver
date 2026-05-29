#!/bin/bash
sed -i '/appnum = world_comm.Get_attr(MPI.APPNUM)/,/local_comm = world_comm.Split(color, 0)/c\
    # Handle MPMD split\
    # We no longer rely on MPI_APPNUM, because Slurm srun with OpenMPI 5 assigns appnum 0 to both components!\
    # Since this script is ALWAYS the DL client, we unconditionally assign it color MPI_UNDEFINED.\
    color = MPI.UNDEFINED\
    local_comm = world_comm.Split(color, 0)' /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/phydll_dl_client.py
