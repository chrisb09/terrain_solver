#include <mpi.h>
#include <iostream>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int* appnum_ptr = nullptr;
    int appnum_flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum_ptr, &appnum_flag);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank " << rank << " appnum_flag: " << appnum_flag << " appnum: " << (appnum_flag ? *appnum_ptr : -1) << std::endl;
    MPI_Finalize();
    return 0;
}
