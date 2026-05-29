#!/bin/bash
sed -i '/int\* appnum_ptr = nullptr;/,/MPI_Comm_split(MPI_COMM_WORLD, color, 0, &local_comm);/c\
    // Participate in the MPMD split to avoid collective mismatches.\
    // We no longer rely on MPI_APPNUM, because Slurm srun with OpenMPI 5 assigns appnum 0 to both components!\
    // Since this binary is ALWAYS the DL client, we unconditionally assign it color MPI_UNDEFINED.\
    const int color = MPI_UNDEFINED;\
    MPI_Comm local_comm = MPI_COMM_NULL;\
    MPI_Comm_split(MPI_COMM_WORLD, color, 0, &local_comm);' /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/dl_client.cpp
