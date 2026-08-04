#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    int len;
    MPI_Get_processor_name(hostname, &len);

    std::cout << "Rank " << rank << " of " << size << " started on host: " << hostname << std::endl;

    if (size < 2) {
        std::cerr << "This test requires at least 2 ranks." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int token = 42;
    if (rank == 0) {
        std::cout << "Rank 0 sending token to Rank 1..." << std::endl;
        MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        std::cout << "Rank 0 token sent successfully." << std::endl;
    } else if (rank == 1) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 1 received token " << token << " from Rank 0!" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "MPI communication test completed successfully." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
