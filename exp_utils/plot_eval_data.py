import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from glob import glob
from pathlib import Path

exp_interp = '../mono-qpd-simple/result/eval/exp_interp'
exp_skipconv_interp = '../mono-qpd-simple/result/eval/exp_skipconv-interp'
exp_fixed_conv = '../mono-qpd-simple/result/eval/exp_fixed-conv'
exp_interp_bs_16 = '/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp_bs-16/'
exp_Interp = '../mono-qpd-simple/result/train/Interp/'

def exp_set_maker(exp_path):
    """exp_name과 sub_dir_name을 데이터프레임에 저장 (pathlib.Path 사용)"""
    
    exp_path = Path(exp_path).resolve()  # 절대 경로로 변환
    exp_name = exp_path.name  # 실험 이름

    sub_dirs = [p for p in exp_path.iterdir() if p.is_dir()]  # 서브 디렉토리 목록
    sub_dir_names = [p.name for p in sub_dirs]  # 서브 디렉토리 이름 목록
    eval_txt_dirs = [p / 'eval_txts.txt' for p in sub_dirs]  # 각 서브 디렉토리 내 eval_txts.txt 파일 경로

    # 존재하지 않는 파일 필터링
    valid_eval_txt_dirs = [p for p in eval_txt_dirs if p.exists()]
    valid_sub_dir_names = [sub_dirs[i].name for i in range(len(sub_dirs)) if eval_txt_dirs[i].exists()]
    valid_sub_dirs = [sub_dirs[i] for i in range(len(sub_dirs)) if eval_txt_dirs[i].exists()]

    # 존재하는 파일이 없으면 빈 데이터프레임 반환
    if not valid_eval_txt_dirs:
        print(f"Error: No valid eval_txts.txt found in {exp_path}")
        return pd.DataFrame()

    # 마커 설정
    marks = [
        'o' if name == 'qpd-test' else
        '^' if name == 'qpd-valid' else
        'v' if name == 'dp-disp' else 'x'
        for name in valid_sub_dir_names
    ]
    
    # 컬러 설정
    if exp_name == 'exp_fixed-conv':
        colors = ['tab:blue'] * len(valid_sub_dirs)
    elif exp_name == 'exp_interp':
        colors = ['tab:red'] * len(valid_sub_dirs)
    elif exp_name == 'exp_skipconv-interp':
        colors = ['tab:green'] * len(valid_sub_dirs)
    else:
        colors = ['tab:purple'] * len(valid_sub_dirs)

    # 저장 경로 설정
    save_paths = [p / 'figures' for p in valid_sub_dirs]

    # 데이터프레임 생성
    df = pd.DataFrame({
        'exp_name': exp_name,
        'sub_dir_name': valid_sub_dir_names,
        'eval_txt_dir': valid_eval_txt_dirs,
        'color': colors,
        'marker': marks,
        'save_path': save_paths
    })
    
    return df

# def exp_set_maker(exp_path):
#     res = {}

#     exp_name = os.path.basename(exp_path)

#     sub_dirs = glob(exp_path + '*')
#     sub_dir_names = [os.path.basename(sub_dir) for sub_dir in sub_dirs]
#     eval_txt_dirs = [os.path.join(sub_dir, 'eval_txts.txt') for sub_dir in sub_dirs]

#     marks = []
#     for sub_dir_name in sub_dir_names:
#         if sub_dir_name == 'qpd-test':
#             marks.append('o')
#         elif sub_dir_name == 'qpd-valid':
#             marks.append('^')
#         elif sub_dir_name == 'dp-disp':
#             marks.append('v')
#         else:
#             marks.append('x')
    
#     colors = ['tab:blue' for i in range(len(sub_dirs))]

#     save_paths = [os.path.join(sub_dir, 'figures/') for sub_dir in sub_dirs]

#     exp_names = [exp_name for i in range(len(sub_dirs))]

#     return list(zip(eval_txt_dirs, exp_names, colors, marks, save_paths))

# def exp_set_maker(exp_path):
#     """exp_name과 sub_dir_name을 key로 하는 딕셔너리를 생성"""
#     exp_name = os.path.basename(exp_path)

#     sub_dirs = glob(exp_path + '*')
#     sub_dir_names = [os.path.basename(sub_dir) for sub_dir in sub_dirs]
#     eval_txt_dirs = [os.path.join(sub_dir, 'eval_txts.txt') for sub_dir in sub_dirs]

#     marks = []
#     for sub_dir_name in sub_dir_names:
#         if sub_dir_name == 'qpd-test':
#             marks.append('o')
#         elif sub_dir_name == 'qpd-valid':
#             marks.append('^')
#         elif sub_dir_name == 'dp-disp':
#             marks.append('v')
#         else:
#             marks.append('x')
    
#     colors = ['tab:blue' for _ in range(len(sub_dirs))]

#     save_paths = [os.path.join(sub_dir, 'figures/') for sub_dir in sub_dirs]

#     # 딕셔너리 생성: (exp_name, sub_dir_name) → (eval_txt_dir, exp_name, color, mark, save_path)
#     exp_dict = {}
#     for sub_dir_name, eval_txt_dir, color, mark, save_path in zip(sub_dir_names, eval_txt_dirs, colors, marks, save_paths):
#         exp_dict[(exp_name, sub_dir_name)] = (eval_txt_dir, exp_name, color, mark, save_path)

#     return exp_dict



# 파일 경로 설정
qpd_test_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-test/eval_txts.txt'
qpd_valid_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp1/qpd-valid/eval_txts.txt'
dp_disp_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp1/dp-disp/eval_txts.txt'
output_dir = '../mono-qpd-simple/result/eval/conv/exp1/'
exp1_qpd_test_sets = (qpd_test_eval_txt_path, 'mono-qpd-resume', 'tab:blue', 'o', '../mono-qpd-simple/result/eval/conv/exp1/qpd-test/figures/')
exp1_qpd_valid_sets = (qpd_valid_eval_txt_path, 'mono-qpd-resume', 'tab:blue', '^', '../mono-qpd-simple/result/eval/conv/exp1/qpd-valid/figures/')
exp1_dp_disp_sets = (dp_disp_eval_txt_path, 'mono-qpd-resume', 'tab:blue', 'v', '../mono-qpd-simple/result/eval/conv/exp1/dp-disp/figures/')
exp1_sets = (exp1_qpd_test_sets, exp1_qpd_valid_sets, exp1_dp_disp_sets)

qpd_test_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp2/qpd-test/eval_txts.txt'
qpd_valid_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp2/qpd-valid/eval_txts.txt'
dp_disp_eval_txt_path = '../mono-qpd-simple/result/eval/conv/exp2/dp-disp/eval_txts.txt'
output_dir = '../mono-qpd-simple/result/eval/conv/exp2/'
exp2_qpd_test_sets = (qpd_test_eval_txt_path, 'mono-qpd-restart', 'tab:red', 'o', '../mono-qpd-simple/result/eval/conv/exp2/qpd-test/figures/')
exp2_qpd_valid_sets = (qpd_valid_eval_txt_path, 'mono-qpd-restart', 'tab:red', '^', '../mono-qpd-simple/result/eval/conv/exp2/qpd-valid/figures/')
exp2_dp_disp_sets = (dp_disp_eval_txt_path, 'mono-qpd-restart', 'tab:red', 'v', '../mono-qpd-simple/result/eval/conv/exp2/dp-disp/figures/')
exp2_sets = (exp2_qpd_test_sets, exp2_qpd_valid_sets, exp2_dp_disp_sets)

qpd_test_eval_txt_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-test/eval_txts.txt'
qpd_valid_eval_txt_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-valid/eval_txts.txt'
dp_disp_eval_txt_path = '../mono-qpd-AiF/result/eval/conv/exp3/qpd-valid/eval_txts.txt'
output_dir = '../mono-qpd-AiF/result/eval/conv/exp3/'
exp3_qpd_test_sets = (qpd_test_eval_txt_path, 'mono-qpd-restart-AiF', 'tab:green', 'o', '../mono-qpd-AiF/result/eval/conv/exp3/qpd-test/figures/')
exp3_qpd_valid_sets = (qpd_valid_eval_txt_path, 'mono-qpd-restart-AiF', 'tab:green', '^', '../mono-qpd-AiF/result/eval/conv/exp3/qpd-valid/figures/')
exp3_sets = (exp3_qpd_test_sets, exp3_qpd_valid_sets)



def plot_by_dataset(exp_df, save_dir='result/figures/'):
    """
    같은 sub_dir_name을 공유하는 실험들을 같은 figure에 플롯하는 함수.
    같은 메트릭에 대해 하나의 figure를 생성.
    
    :param exp_df: exp_set_maker()로 만든 DataFrame (columns: exp_name, sub_dir_name, eval_txt_dir, color, marker, save_dir)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # 전체 그래프 저장 폴더 생성

    # sub_dir_name 기준으로 그룹화
    grouped_exp = exp_df.groupby('sub_dir_name')

    # 각 sub_dir_name 그룹별로 그래프 생성
    for sub_dir_name, group in grouped_exp:
        metric_data = {}  # 같은 메트릭을 모아 저장할 딕셔너리

        for _, row in group.iterrows():
            eval_txt_dir, exp_name, color, marker = row['eval_txt_dir'], row['exp_name'], row['color'], row['marker']

            # 파일 읽기
            try:
                df = pd.read_csv(eval_txt_dir, sep='\s+')
            except pd.errors.EmptyDataError:
                print(f"Error: EmptyDataError in {eval_txt_dir}")
                continue
            
            # Epochs 추출
            df['epochs'] = df['model_epochs'].str.extract(r'(\d+)_epoch').astype(int)
            df = df.sort_values(by='epochs')

            # Epoch이 0인 행 제거
            df = df[df['epochs'] != 0]

            # 플롯할 컬럼 식별 (model_epochs, epochs 제외)
            metrics = set(df.columns) - {'model_epochs', 'epochs'}

            # 같은 메트릭을 저장할 딕셔너리 초기화
            for col in metrics:
                if col not in metric_data:
                    metric_data[col] = []
                metric_data[col].append((df['epochs'], df[col], exp_name, color, marker))

        # 같은 sub_dir_name에 대해 각 메트릭별 그래프 생성
        for metric, data_list in metric_data.items():
            plt.figure(figsize=(10, 6))

            for epochs, values, exp_name, color, marker in data_list:
                plt.plot(epochs, values, label=f"{exp_name.replace('exp_', '')}", marker=marker, color=color)

            plt.title(f"{sub_dir_name} - {metric}")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)

            # 그래프 저장
            save_path = save_dir / f"{sub_dir_name}/figures/{metric}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)  # 서브 디렉토리 생성

            plt.savefig(save_path)
            plt.close()

        print(f"{sub_dir_name} figures saved")

def plot_metrics(file_path, label, color, marker, output_dir):
    """지정된 파일 경로에서 메트릭을 플롯하여 저장하는 함수"""
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성

    # 파일 읽기
    df = pd.read_csv(file_path, sep='\s+')

    # Epochs 추출
    try:
        df['epochs'] = df['model_epochs'].str.extract(r'(\d+)_epoch').astype(int)
    except ValueError as e:
        print(f"Error: {e}")
        # print(f"model_epochs: {df['model_epochs']}")
        print(f"file_path: {file_path}")
        return

    # Epochs 순서대로 정렬
    df = df.sort_values(by='epochs')

    # Epoch이 0인 행 제거
    df = df[df['epochs'] != 0]

    # 모든 컬럼 식별 (model_epochs, epochs 제외)
    metrics = set(df.columns) - {'model_epochs', 'epochs'}

    # 각 메트릭별 그래프 저장
    for col in metrics:
        plt.figure()
        plt.plot(df['epochs'], df[col], label=label, marker=marker, color=color)
        plt.title(f'{label} {col} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(col)
        plt.legend()
        plt.savefig(f'{output_dir}{col}.png')
        plt.close()

def plot_metrics_comparison_common_epochs(file_path1, file_path2, label1, label2, color1, color2, marker1, marker2, output_dir):
    """두 개의 파일에서 공통 epoch 범위 내에서 메트릭을 비교하여 플롯하는 함수"""
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성

    # 파일 읽기
    df1 = pd.read_csv(file_path1, delim_whitespace=True)
    df2 = pd.read_csv(file_path2, delim_whitespace=True)

    # Epochs 추출
    df1['epochs'] = df1['model_epochs'].str.extract(r'(\d+)_epoch').astype(int)
    df2['epochs'] = df2['model_epochs'].str.extract(r'(\d+)_epoch').astype(int)

    # 공통 epoch의 최대값 계산
    max_common_epoch = min(df1['epochs'].max(), df2['epochs'].max())

    # 공통 epoch 범위 내에서 데이터 필터링
    df1 = df1[df1['epochs'] <= max_common_epoch]
    df2 = df2[df2['epochs'] <= max_common_epoch]

    # 공통 컬럼 식별
    common_columns = set(df1.columns) & set(df2.columns) - {'model_epochs', 'epochs'}

    # 공통 메트릭 그래프 저장
    for col in common_columns:
        plt.figure()
        plt.plot(df1['epochs'], df1[col], label=label1, marker=marker1, color=color1)
        plt.plot(df2['epochs'], df2[col], label=label2, marker=marker2, color=color2)
        plt.title(f'{col} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(col)
        plt.legend()
        plt.savefig(f'{output_dir}/{col}_comparison.png')
        plt.close()

    print(f"Plots saved up to max common epoch: {max_common_epoch}")

if __name__ == "__main__":
    # ---- User Input ---- 사용할 데이터셋 목록
    # exp_sets = exp1_sets + exp2_sets + exp3_sets
    # exp_sets = exp_set_maker(exp_fixed_conv)
    # exp_sets = {**exp_sets, **exp_set_maker(exp_interp)}
    # exp_sets = {**exp_sets, **exp_set_maker(exp_skipconv_interp)}

    # df_fixed_conv = exp_set_maker(exp_fixed_conv)
    # df_interp = exp_set_maker(exp_interp)
    # df_skipconv_interp = exp_set_maker(exp_skipconv_interp)
    # exp_sets = pd.concat([df_fixed_conv, df_interp, df_skipconv_interp], ignore_index=True)
    # exp_sets = df_interp
    # df_interp_bs_16 = exp_set_maker(exp_interp_bs_16)
    # exp_sets = df_interp_bs_16

    exp_sets = exp_set_maker(exp_Interp)
    exp_sets = pd.concat([exp_sets])
    # ---- User Input End ----
    plot_by_dataset(exp_sets)