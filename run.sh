# method: 0->sndt, 1->ndtd2d, 2->sicp
# 720, 960, 965
# SNDT
devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log24/vepcs.ser -o /tmp/path.ser -m 0
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log24/gt.ser -o ../Analysis/20210805/results/pc/sndt/pc_sndt_log24

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-1/vepcs.ser -o /tmp/path.ser -m 0
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-1/gt.ser -o ../Analysis/20210805/results/pc/sndt/pc_sndt_log62-1

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-2/vepcs.ser -o /tmp/path.ser -m 0
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-2/gt.ser -o ../Analysis/20210805/results/pc/sndt/pc_sndt_log62-2

# NDT-D2D
devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log24/vepcs.ser -o /tmp/path.ser -m 1
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log24/gt.ser -o ../Analysis/20210805/results/pc/ndtd2d/pc_ndtd2d_log24

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-1/vepcs.ser -o /tmp/path.ser -m 1
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-1/gt.ser -o ../Analysis/20210805/results/pc/ndtd2d/pc_ndtd2d_log62-1

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-2/vepcs.ser -o /tmp/path.ser -m 1
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-2/gt.ser -o ../Analysis/20210805/results/pc/ndtd2d/pc_ndtd2d_log62-2

# SICP
devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log24/vepcs.ser -o /tmp/path.ser -m 2
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log24/gt.ser -o ../Analysis/20210805/results/pc/sicp/pc_sicp_log24

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-1/vepcs.ser -o /tmp/path.ser -m 2
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-1/gt.ser -o ../Analysis/20210805/results/pc/sicp/pc_sicp_log62-1

devel/lib/sndt_exec/test_path -i ../Analysis/1Data/log62-2/vepcs.ser -o /tmp/path.ser -m 2
devel/lib/tool/pathtotext -e /tmp/path.ser -g ../Analysis/1Data/log62-2/gt.ser -o ../Analysis/20210805/results/pc/sicp/pc_sicp_log62-2

src/rpg_te/scripts/analyze_trajectories.py --results_dir ../Analysis/20210805/results --output_dir ../Analysis/20210805/results/wmc --dataset log24,log62-1,log62-2 --odometry_error_per_dataset --overall_odometry_error --plot_trajectories --rmse_table --png --recalculate_errors
