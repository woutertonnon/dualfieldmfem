#!/bin/bash
#SBATCH --job-name=stokes_mg_cvg
#SBATCH --time=04:00:00              # Time limit set to 4 hours
#SBATCH --ntasks=1                   # 1 task for the python script
#SBATCH --cpus-per-task=32           # 32 allocated cores
#SBATCH --mem-per-cpu=2G             # 2 GB memory per core
#SBATCH --output=mg_cvg_log_%j.out   # Standard output log
#SBATCH --error=mg_cvg_log_%j.err    # Standard error log

# 1. Load the current Euler software stack and required modules
module load stack/2024-06
module load python/3.12.8
module load gcc/12.2.0            
module load cmake
module load openmpi
module load boost
module load eigen
module load metis # Completely useless here, but needs to be loaded for suite-sparse
module load suite-sparse
module load spectra
module load doxygen

# 2. Build the C++ executable (Purge old build first)
# echo "Purging old release directory and starting compilation..."
# rm -rf ../release
# mkdir -p ../release
# cd ../release
# cmake -DCMAKE_BUILD_TYPE=Release ..

cd ../build

make -j32                         # Compile in parallel using all 32 cores
echo "Compilation finished."

# 3. Return to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# 4. Set environment variables
export OMP_NUM_THREADS=2

# 5. Execute the python script
echo "Starting Python parameter study..."
python mg_conv.py
