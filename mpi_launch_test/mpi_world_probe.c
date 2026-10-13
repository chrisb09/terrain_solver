/*
 * Minimal MPI launch probe for validating multi-node process placement and
 * shared-MPI-world connectivity, independent of the terrain solver.
 *
 * Usage: mpi_world_probe [role]
 *
 * Prints one line per rank: rank, world size, hostname, role. Rank 0 then
 * prints the full gathered world table so we can confirm that processes on
 * both nodes belong to a single MPI_COMM_WORLD.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = -1, size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);

    const char *role = (argc > 1) ? argv[1] : "task";

    printf("PROBE rank=%d/%d host=%s role=%s\n", rank, size, host, role);
    fflush(stdout);

    /* Gather hostnames to prove all ranks share one communicator. */
    char *all = NULL;
    if (rank == 0)
        all = (char *)malloc((size_t)size * 256);
    MPI_Gather(host, 256, MPI_CHAR, all, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== WORLD size=%d ===\n", size);
        for (int i = 0; i < size; ++i)
            printf("  world[%d] = %s\n", i, all + (size_t)i * 256);
        fflush(stdout);
        free(all);
        printf("=== PROBE_OK size=%d ===\n", size);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
