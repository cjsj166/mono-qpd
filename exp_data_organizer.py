from exp_utils.aggregate_eval_results import *
from exp_utils.plot_eval_data import *
from exp_args_settings.train_settings import get_train_config
import argparse

exp_Interp = '../mono-qpd-simple/result/train/Interp/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='Interp', required=True, help="name your experiment")
    args = parser.parse_args()
    
    # get the train config
    conf = get_train_config(args.exp_name)

    exp_test_sets = select_sub_dir_and_save_path(conf.save_path)
    
    for exp_dir_path, exp_save_path in exp_test_sets:
        dir_paths = exp_dir_path.glob('*_epoch_*')
        aggregate_eval_results(dir_paths, exp_save_path)

    exp_sets = exp_set_maker(conf.save_path)
    exp_sets = pd.concat([exp_sets])
    
    plot_by_dataset(exp_sets, conf.save_path)