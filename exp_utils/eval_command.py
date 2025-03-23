
import os

exp1_checkpoint_paths = [
        'result/train/20250205_043932_exp1_40epochs/checkpoints/10_epoch_15051_Mono-QPD.pth',
        'result/train/20250205_043932_exp1_40epochs/checkpoints/20_epoch_30101_Mono-QPD.pth',
        'result/train/20250205_043932_exp1_40epochs/checkpoints/30_epoch_45151_Mono-QPD.pth',
        'result/train/20250205_043932_exp1_40epochs/checkpoints/40_epoch_60201_Mono-QPD.pth',
        'result/train/20240205_153455_exp1_132epochs/checkpoints/50_epoch_75250_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/60_epoch_90300_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/70_epoch_105350_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/80_epoch_120400_Mono-QPD.pth',
        'result/train/20240205_153455_exp1_132epochs/checkpoints/90_epoch_135450_Mono-QPD.pth',
        'result/train/20240205_153455_exp1_132epochs/checkpoints/100_epoch_150500_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/110_epoch_165550_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/120_epoch_180600_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/130_epoch_195650_Mono-QPD.pth', 
        'result/train/20240205_153455_exp1_132epochs/checkpoints/132_epoch_198660_Mono-QPD.pth', 
    ]
exp1_project_name = 'mono-qpd-simple'

exp1_checkpoint_paths_left = [
        'result/train/20250208_202757_exp1_140epochs/checkpoints/135_epoch_203175_Mono-QPD.pth',
        'result/train/20250208_202757_exp1_140epochs/checkpoints/140_epoch_210700_Mono-QPD.pth',    
        'result/train/20250209_000402_exp1_265epochs/checkpoints/145_epoch_218225_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/150_epoch_225750_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/155_epoch_233275_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/160_epoch_240800_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/165_epoch_248325_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/170_epoch_255850_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/175_epoch_263375_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/180_epoch_270900_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/185_epoch_278425_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/190_epoch_285950_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/195_epoch_293475_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/200_epoch_301000_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/205_epoch_308525_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/210_epoch_316050_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/215_epoch_323575_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/220_epoch_331100_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/225_epoch_338625_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/230_epoch_346150_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/235_epoch_353675_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/240_epoch_361200_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/245_epoch_368725_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/250_epoch_376250_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/255_epoch_383775_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/260_epoch_391300_Mono-QPD.pth',
        'result/train/20250209_000402_exp1_265epochs/checkpoints/265_epoch_398825_Mono-QPD.pth'
]


exp2_checkpoint_paths = [
    'result/train/20250208_012006_exp2_99epochs/checkpoints/5_epoch_7525_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/10_epoch_15050_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/15_epoch_22575_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/20_epoch_30100_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/25_epoch_37625_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/30_epoch_45150_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/35_epoch_52675_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/40_epoch_60200_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/45_epoch_67725_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/50_epoch_75250_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/55_epoch_82775_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/60_epoch_90300_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/65_epoch_97825_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/70_epoch_105350_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/75_epoch_112875_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/80_epoch_120400_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/85_epoch_127925_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/90_epoch_135450_Mono-QPD.pth',
    'result/train/20250208_012006_exp2_99epochs/checkpoints/95_epoch_142975_Mono-QPD.pth'
    ]
exp2_project_name = 'mono-qpd-simple'

exp3_checkpoint_paths = [
    'result/train/20250209_024242_exp3_125epochs/checkpoints/5_epoch_7525_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/10_epoch_15050_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/15_epoch_22575_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/20_epoch_30100_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/25_epoch_37625_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/30_epoch_45150_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/35_epoch_52675_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/40_epoch_60200_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/45_epoch_67725_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/50_epoch_75250_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/55_epoch_82775_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/60_epoch_90300_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/65_epoch_97825_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/70_epoch_105350_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/75_epoch_112875_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/80_epoch_120400_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/85_epoch_127925_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/90_epoch_135450_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/95_epoch_142975_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/100_epoch_150500_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/105_epoch_158025_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/110_epoch_165550_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/115_epoch_173075_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/120_epoch_180600_Mono-QPD.pth',
    'result/train/20250209_024242_exp3_125epochs/checkpoints/125_epoch_188125_Mono-QPD.pth'
    ]
exp3_project_name = 'mono-qpd-AiF'

if __name__ == "__main__":
    # ---- User input ----
    checkpoint_paths = exp1_checkpoint_paths
    experiment_name = 'exp1'
    project_name = exp1_project_name
    # ---- User input end ----
    
    print("cd /mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple;")
    print("conda activate mono-qpd;")

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)[:-4]
        print(f"python evaluate_mono_qpd.py --restore_ckpt {checkpoint_path} --datasets_path datasets/QP-Data --save_path result/eval/conv/{experiment_name}/qpd-test/{checkpoint_name} --feature_converter conv --save_result False;")
        if experiment_name != 'exp3':
            print(f"python evaluate_mono_qpd.py --restore_ckpt {checkpoint_path} --datasets_path datasets/MDD_dataset --dataset MDD --save_path result/eval/conv/{experiment_name}/dp-disp/{checkpoint_name} --feature_converter conv --save_result False;")
    
        
    # python evaluate_mono_qpd.py --restore_ckpt result/train/20240205_153455/checkpoints/132_epoch_198660_Mono-QPD.pth --datasets_path datasets/MDD_dataset --dataset MDD --save_path result/eval/conv/dp-disp/132_epoch_198660_Mono-QPD --feature_converter conv
