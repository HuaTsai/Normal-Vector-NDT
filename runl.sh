execfile=devel/lib/sndt_exec/testl_path
folder=resultslrt
Date=20210816
Analysis=src/rpg_te/scripts/analyze_trajectories.py

echo "log24"
$execfile -d log24 -o ../Analysis/$Date/$folder

echo "log35-1"
$execfile -d log35-1 -o ../Analysis/$Date/$folder

echo "log62-1"
$execfile -d log62-1 -o ../Analysis/$Date/$folder

echo "log62-2"
$execfile -d log62-2 -o ../Analysis/$Date/$folder

$Analysis --results_dir ../Analysis/$Date/$folder \
          --output_dir ../Analysis/$Date/$folder \
          --dataset log24,log35-1,log62-1,log62-2 \
          --odometry_error_per_dataset --overall_odometry_error \
          --plot_trajectories --rmse_table --png --recalculate_errors
