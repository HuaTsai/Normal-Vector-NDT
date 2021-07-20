devel/lib/sndt_exec/test_path -i ../Analysis/1Data/vepcs24.ser -o ../Analysis/20210719/paths/est_log24.ser -u
devel/lib/sndt_exec/test_path -i ../Analysis/1Data/vepcs62-1.ser -o ../Analysis/20210719/paths/est_log62-1.ser -u
devel/lib/sndt_exec/test_path -i ../Analysis/1Data/vepcs62-2.ser -o ../Analysis/20210719/paths/est_log62-2.ser -u

devel/lib/tool/pathtotext -e ../Analysis/20210719/paths/est_log24.ser -g ../Analysis/20210719/paths/gt_log24.ser -o ../Analysis/20210719/results/pc/sndt/pc_sndt_log24
devel/lib/tool/pathtotext -e ../Analysis/20210719/paths/est_log62-1.ser -g ../Analysis/20210719/paths/gt_log62-1.ser -o ../Analysis/20210719/results/pc/sndt/pc_sndt_log62-1
devel/lib/tool/pathtotext -e ../Analysis/20210719/paths/est_log62-2.ser -g ../Analysis/20210719/paths/gt_log62-2.ser -o ../Analysis/20210719/results/pc/sndt/pc_sndt_log62-2

src/rpg_te/scripts/analyze_trajectories.py --results_dir ../Analysis/20210719/results --output_dir ../Analysis/20210719/results --dataset log24,log62-1,log62-2 --odometry_error_per_dataset --overall_odometry_error --plot_trajectories --rmse_table --png
