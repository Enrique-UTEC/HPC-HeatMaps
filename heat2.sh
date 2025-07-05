#!/bin/bash
#SBATCH -J heat_mpi        # Nombre del job
#SBATCH -p standard
#SBATCH -c 1
#SBATCH --ntasks=32        # Máximo tareas MPI que probarás
#SBATCH --time=01:00:00

module load mpich/3.4.3-ofi

# Cabecera limpia
echo "N,Procesos,Total_s,Calculo_s,Comunicacion_s" > metrics.txt

# Lista de N y P a barrer
for N in 512 1024 2048 4096 8192; do
    for PROCS in 1 2 4 8 16 32; do
        # para weak-scaling podrías forzar N ∝ sqrt(PROCS)
        echo "Ejecutando N=$N con P=$PROCS..."
        mpirun -np $PROCS ./heat2d_mpi $N
    done
done
