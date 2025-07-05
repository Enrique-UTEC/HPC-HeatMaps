#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------
// Parámetros físicos por defecto
// ---------------------------
int N = 1000;           // Tamaño por defecto de la grilla NxN
#define STEPS 500       // Número de pasos de tiempo
#define ALPHA 0.01      // Difusividad térmica
#define DX 1.0          // Paso espacial en x
#define DY 1.0          // Paso espacial en y
#define DT 0.1          // Paso temporal

// ---------------------------
// Función: boundary_condition
// Devuelve la temperatura fija en los bordes
// ---------------------------
double boundary_condition(int i, int j) {
    return 100.0;
}

// ---------------------------
// Función: initialize
// Inicializa la subgrilla de cada proceso, incluyendo ghost-rows
// ---------------------------
void initialize(double* u, int local_rows, int cols, int rank, int size) {
    int base = N / size;
    int rem  = N % size;
    int start_row;
    if (rank < rem) {
        start_row = rank * (base + 1);
    } else {
        start_row = rem * (base + 1) + (rank - rem) * base;
    }

    // ghost-row superior (frontera global i=0)
    if (rank == 0) {
        for (int j = 0; j < cols; j++) {
            u[j] = boundary_condition(0, j);
        }
    }

    // ghost-row inferior (frontera global i=N-1)
    if (rank == size - 1) {
        int last = local_rows - 1;
        for (int j = 0; j < cols; j++) {
            u[last * cols + j] = boundary_condition(N - 1, j);
        }
    }

    // inicializar filas interiores
    for (int i = 1; i < local_rows - 1; i++) {
        int global_i = start_row + (i - 1);
        for (int j = 0; j < cols; j++) {
            if (global_i == 0 || global_i == N - 1 || j == 0 || j == cols - 1) {
                u[i * cols + j] = boundary_condition(global_i, j);
            } else {
                u[i * cols + j] = 0.0;
            }
        }
    }
}

// ---------------------------
// Función: exchange
// Intercambia ghost rows (superior e inferior) y mide tiempo de comunicación
// ---------------------------
void exchange(double* u, int local_rows, int cols, int rank, int size, double* comm_time) {
    MPI_Status status;
    double t0 = MPI_Wtime();

    if (rank > 0) {
        MPI_Sendrecv(u + cols, cols, MPI_DOUBLE, rank - 1, 0,
                     u,            cols, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) {
        MPI_Sendrecv(u + (local_rows - 2) * cols, cols, MPI_DOUBLE, rank + 1, 0,
                     u + (local_rows - 1) * cols, cols, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }

    double t1 = MPI_Wtime();
    *comm_time += (t1 - t0);
}

// ---------------------------
// Función: update
// Actualiza temperaturas usando diferencias finitas explícitas y mide tiempo de cómputo
// ---------------------------
void update(double* u, double* u_new, int local_rows, int cols, double* comp_time) {
    double t0 = MPI_Wtime();

    for (int i = 1; i < local_rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int idx = i * cols + j;
            double dudx2 = (u[idx - 1]    - 2 * u[idx] + u[idx + 1]) / (DX * DX);
            double dudy2 = (u[idx - cols] - 2 * u[idx] + u[idx + cols]) / (DY * DY);
            u_new[idx] = u[idx] + ALPHA * DT * (dudx2 + dudy2);
        }
    }

    double t1 = MPI_Wtime();
    *comp_time += (t1 - t0);
}

// ---------------------------
// Función principal: main
// ---------------------------
int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    int base_rows = N / size;
    int remainder = N % size;
    int local_rows = base_rows + 2 + (rank < remainder ? 1 : 0);

    double* u     = calloc(local_rows * N, sizeof(double));
    double* u_new = calloc(local_rows * N, sizeof(double));

    initialize(u, local_rows, N, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double time_start = MPI_Wtime();

    double comm_time = 0.0;
    double comp_time = 0.0;

    for (int t = 0; t < STEPS; t++) {
        exchange(u, local_rows, N, rank, size, &comm_time);
        update(u, u_new, local_rows, N, &comp_time);
        double* tmp = u; u = u_new; u_new = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double time_end = MPI_Wtime();
    double total_time = time_end - time_start;

    if (rank == 0) {
        printf("[Resultados] N=%d, P=%d -> Total: %f s, Cálculo: %f s, Comunicación: %f s\n",
               N, size, total_time, comp_time, comm_time);
        FILE* f = fopen("metrics.txt", "a");
        if (f) {
            fprintf(f, "%d,%d,%.6f,%.6f,%.6f\n",
                    N, size, total_time, comp_time, comm_time);
            fclose(f);
        }
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}
