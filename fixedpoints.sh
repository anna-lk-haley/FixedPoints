#!/bin/bash

########SBATCH -A ctb-steinman
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
#SBATCH --time=23:59:00
######SBATCH --time=1:00:00
######SBATCH -p debug
#SBATCH --job-name fixedpoints
#SBATCH --output=fixedpoints_%j.txt
#SBATCH --mail-type=END

export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/scratch/s/steinman/ahaleyyy/.config/mpl
export PYVISTA_USERDATA_PATH=/scratch/s/steinman/ahaleyyy/.local/share/pyvista
export XDG_RUNTIME_DIR=/scratch/s/steinman/ahaleyyy/.local/temp
export TEMPDIR=$SCRATCH/.local/temp
export TMPDIR=$SCRATCH/.local/temp
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_PANEL=true

module load CCEnv StdEnv/2020 gcc/9.3.0 vtk/9.0.1 python/3.7.7 
source $HOME/.virtualenvs/toolsenv/bin/activate

results_folder='../mesh_rez/cases/case_A/case_028_low/results/art_PTSeg028_low_I1_FC_VENOUS_Q557_Per915_Newt370_ts15660_cy2_uO1'
wss_folder='$SCRATCH/mesh_rez/cases/case_A/case_028_low/results/art_PTSeg028_low_I1_FC_VENOUS_Q557_Per915_Newt370_ts15660_cy2_uO1/wss_files'
wss_files=$(eval "ls -l $wss_folder | wc -l")
echo There are $wss_files WSS files
grps=$(($wss_files/39)) #39 processes - rounded down number
rem=$(($wss_files- $grps*39)) #put the remaining files on the final proc
num_files=0
for ((i=0 ; i < 39 ; i++ ))
do
    start_file=$(($num_files))
    num_files=$((($i+ 1)*$grps))
    end_file=$(($num_files-1))
    (~/../macdo708/xvfb-run-safe python fixedpoints.py $results_folder PTSeg028_low $start_file $end_file && echo finished $i th group)&
done
start_file=$num_files
end_file=$(($num_files+$rem-1))
(~/../macdo708/xvfb-run-safe python fixedpoints.py $results_folder PTSeg028_low $start_file $end_file && echo 'finished 40 th group')&
wait

#~/../macdo708/xvfb-run-safe python fixedpoints.py $results_folder PTSeg028_low 0 $(($wss_files-1))