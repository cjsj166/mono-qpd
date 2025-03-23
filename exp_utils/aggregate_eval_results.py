import os
from glob import glob
from pathlib import Path

# exp 1
# qpd-test
exp1_qpd_test_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-test/'
exp1_qpd_test_save_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-test/eval_txts.txt'
exp1_qpd_test_sets = [exp1_qpd_test_path, exp1_qpd_test_save_path]

# qpd-valid
exp1_qpd_valid_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-valid/'
exp1_qpd_valid_save_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-valid/eval_txts.txt'
exp1_qpd_valid_sets = [exp1_qpd_valid_path, exp1_qpd_valid_save_path]

# dp-disp
exp1_dp_disp_path = '../mono-qpd-simple/result/eval/conv/exp1/dp-disp/'
exp1_dp_disp_save_path = '../mono-qpd-simple/result/eval/conv/exp1/dp-disp/eval_txts.txt'
exp1_dp_disp_sets = [exp1_dp_disp_path, exp1_dp_disp_save_path]

exp1_sets = [exp1_qpd_test_sets, exp1_qpd_valid_sets, exp1_dp_disp_sets]

# exp 2
# qpd-test
exp2_train_dir = '../mono-qpd-simple/result/eval/conv/exp2/qpd-test/'
exp2_qpd_test_save_path = '../mono-qpd-simple/result/eval/conv/exp2/qpd-test/eval_txts.txt'
exp2_qpd_test_sets = [exp2_train_dir, exp2_qpd_test_save_path]

# qpd-valid
exp2_qpd_valid_path = '../mono-qpd-simple/result/eval/conv/exp2/qpd-valid/'
exp2_qpd_valid_save_path = '../mono-qpd-simple/result/eval/conv/exp2/qpd-valid/eval_txts.txt'
exp2_qpd_valid_sets = [exp2_qpd_valid_path, exp2_qpd_valid_save_path]

# dp-disp
exp2_train_dir = '../mono-qpd-simple/result/eval/conv/exp2/dp-disp/'
exp2_dp_disp_save_path = '../mono-qpd-simple/result/eval/conv/exp2/dp-disp/eval_txts.txt'
exp2_dp_disp_sets = [exp2_train_dir, exp2_dp_disp_save_path]

exp2_sets = [exp2_qpd_test_sets, exp2_qpd_valid_sets, exp2_dp_disp_sets]

# exp3
# qpd-test
exp3_train_dir = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-test/'
exp3_qpd_test_save_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-test/eval_txts.txt'
exp3_qpd_test_sets = [exp3_train_dir, exp3_qpd_test_save_path]

# qpd-valid
exp3_qpd_valid_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-valid/'
exp3_qpd_valid_save_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-valid/eval_txts.txt'
exp3_qpd_valid_sets = [exp3_qpd_valid_path, exp3_qpd_valid_save_path]

exp3_sets = [exp3_qpd_test_sets, exp3_qpd_valid_sets]

exp_fixed_conv = '../mono-qpd-simple/result/eval/exp_fixed-conv/'
exp_interp = '../mono-qpd-simple/result/eval/exp_interp/'
exp_skipconv_interp = '../mono-qpd-simple/result/eval/exp_skipconv-interp/'

exp_interp_bs_16 = '/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp_bs-16/'

exp_Interp = '../mono-qpd-simple/result/train/Interp/'

def aggregate_eval_results(eval_txt_dirs, save_path):

    eval_txts = []
    for eval_txt_dir in eval_txt_dirs:

        found_eval_results = sorted(glob(os.path.join(eval_txt_dir, '*.txt'))) # check if the latest file exists

        if found_eval_results:
            eval_txts.append(found_eval_results[-1])
        
        # if glob_path:
        #     eval_txts = [sorted(glob(os.path.join(eval_txt_dir, '*.txt')))[-1] for eval_txt_dir in eval_txt_dirs]
    
    print('')
    common_metrics = ''
    
    min_epe_epoch = {'epe':1000, 'epoch':0}
    min_rmse_epoch = {'rmse':1000, 'epoch':0}
    min_ai1_epoch = {'ai1':1000, 'epoch':0}
    min_ai2_epoch = {'ai2':1000, 'epoch':0}

    with save_path.open('w') as f2:
        for eval_txt in eval_txts:
            with open(eval_txt, 'r') as f1:
                lines = f1.readlines()

                metrics = lines[0].replace('filename', 'model_epochs')
                metrics_list = metrics.split()
                epoch_position = metrics_list.index('model_epochs')

                if 'epe' in metrics_list:
                    epe_position = metrics_list.index('epe')
                if 'rmse' in metrics_list:
                    rmse_position = metrics_list.index('rmse')
                if 'ai1' in metrics_list:
                    ai1_position = metrics_list.index('ai1')
                if 'ai2' in metrics_list:
                    ai2_position = metrics_list.index('ai2')
                
                if metrics != common_metrics:
                    f2.write(metrics)
                    common_metrics = metrics
                    
                line_to_write = lines[1].replace('mean', eval_txt.split('/')[-2])


                # find min epe and rmse
                line_to_write_list = line_to_write.split()
                epoch = int(line_to_write_list[epoch_position].split('_')[0])
                if 'epe' in metrics_list:
                    epe = float(line_to_write_list[epe_position])
                    if epe < min_epe_epoch['epe']:
                        min_epe_epoch['epe'] = epe
                        min_epe_epoch['epoch'] = epoch
                if 'rmse' in metrics_list:
                    rmse = float(line_to_write_list[rmse_position])
                    if rmse < min_rmse_epoch['rmse']:
                        min_rmse_epoch['rmse'] = rmse
                        min_rmse_epoch['epoch'] = epoch
                if 'ai1' in metrics_list:
                    ai1 = float(line_to_write_list[ai1_position])
                    if ai1 < min_ai1_epoch['ai1']:
                        min_ai1_epoch['ai1'] = ai1
                        min_ai1_epoch['epoch'] = epoch
                if 'ai2' in metrics_list:
                    ai2 = float(line_to_write_list[ai2_position])
                    if ai2 < min_ai2_epoch['ai2']:
                        min_ai2_epoch['ai2'] = ai2
                        min_ai2_epoch['epoch'] = epoch


                f2.write(line_to_write)
                # print(line_to_write)
        
    save_path = Path(save_path)
    print(f'min metrics on {save_path.parts[-2]}')
    print(f'min epe: {min_epe_epoch}')    
    print(f'min rmse: {min_rmse_epoch}')
    print(f'min ai1: {min_ai1_epoch}')
    print(f'min ai2: {min_ai2_epoch}')
    print(f'metrics are aggregated and saved as {save_path}')
    
    return

def select_sub_dir_and_save_path(exp_dir_path):
    valid_dirs = ['qpd-valid', 'qpd-test', 'dpd-disp', 'dp-disp']
    exp_dir_path = Path(exp_dir_path)
    sub_dir_paths = exp_dir_path.glob('*')
    sub_dir_paths = [sub_dir_path for sub_dir_path in sub_dir_paths if sub_dir_path.name in valid_dirs]
    # sub_dir_paths = glob(os.path.join(exp_dir_path, '*'))
    
    save_paths = [sub_dir_path / 'eval_txts.txt' for sub_dir_path in sub_dir_paths]
        
    return list(zip(sub_dir_paths, save_paths))

if __name__ == "__main__":
    # ---- user input ---- list of datasets where we aggregate eval results.
    # exp_test_sets = exp1_sets + exp2_sets + exp3_sets
    # exp_test_sets = select_sub_dir_and_save_path(exp_skipconv_interp) + select_sub_dir_and_save_path(exp_interp) + select_sub_dir_and_save_path(exp_fixed_conv)
    # exp_test_sets = select_sub_dir_and_save_path(exp_interp_bs_16)
    exp_test_sets = select_sub_dir_and_save_path(exp_Interp)
    # ---- user input end ----
    
    for exp_dir_path, exp_save_path in exp_test_sets:
        dir_paths = exp_dir_path.glob('*_epoch_*')
        aggregate_eval_results(dir_paths, exp_save_path)
        