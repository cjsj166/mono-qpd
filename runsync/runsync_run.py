import runpy
import sys
import argparse
from presets import get_run_setting
import os
import debugpy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def dataclass_to_argv(dclass):
    """
    Convert a dataclass to a list of command line arguments.
    """
    args = []
    for field in dclass.__dataclass_fields__.keys():
        value = getattr(dclass, field)
        args.append(f'--{field}')
        args.append(str(value))
    return args

if __name__ == "__main__":
    # debugpy.listen(("0.0.0.0", 5678))  # 모든 인터페이스에서 수신 가능
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()

    parser = argparse.ArgumentParser('Script Composer')

    # run_script path
    parser.add_argument('--run_script_name',default='train.py',help=".py file to run")

    # run setting
    parser.add_argument('--run_setting_name',default='BaseConfig',help="arguments to run script")

    # custom arguments
    args = parser.parse_args()

    run_setting = get_run_setting(args.run_setting_name)
    sys.argv[1:] = dataclass_to_argv(run_setting)

    # os.chdir('..')

    runpy.run_path(args.run_script_name, run_name="__main__")