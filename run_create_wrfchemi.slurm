#!/bin/bash

#SBATCH --job-name=PythonEmission           # nom du job
#SBATCH --partition=zen4                    # Nom d'une partition pour une exÃ©tion cpu
#SBATCH --ntasks=1                          # nombre de taches
#SBATCH --ntasks-per-node=1                 # nombre de taches MPI par noeud
#SBATCH --mem=10GB                          # memory limit
#SBATCH --time=24:00:00                     # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=PythonEmission%j.out       # nom du fichier de sortie
#SBATCH --error=PythonEmission%j.out        # nom du fichier d'erreur (ici en commun avec la sortie)

module purge
module load python/meso-3.9
### conda activate my_python3
### module load netcdf4/4.3.3.1-ifort
### export PROJ_LIB='/home/onishi/.conda/pkgs/proj4-5.2.0-h14c3975_1001/share/proj/'

### rm PythonEmission*.out

which python
mpirun python ./python_temp.py3
