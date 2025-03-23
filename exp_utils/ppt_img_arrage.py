import os
from glob import glob
import shutil

real_qpd_a = [ # center, interp, skipconv-interp, qpdnet
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/conv/exp1/qpd-real/qpd-valid-best/real-qpd-test/375_epoch_564375_Mono-QPD/src/Dataset_A_1218/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp/real-qpd-test/115_epoch_173075_Mono-QPD/est/A/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_skipconv-interp/real-qpd/qpd-valid-best/real-qpd-test/100_epoch_150500_Mono-QPD/est/A/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/QPDNet/result/eval/real-qpd/est/Dataset_A_1218/scale3/test_c/source/seq_1",
    ]

real_qpd_b = [
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/conv/exp1/qpd-real/qpd-valid-best/real-qpd-test/375_epoch_564375_Mono-QPD/src/Dataset_B_1218/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp/real-qpd-test/115_epoch_173075_Mono-QPD/est/B/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_skipconv-interp/real-qpd/qpd-valid-best/real-qpd-test/100_epoch_150500_Mono-QPD/est/B/scale3/test_c/source/seq_1",
        "/mnt/d/Mono+Dual/QPDNet/result/eval/real-qpd/est/Dataset_B_1218/scale3/test_c/source/seq_1",
    ]

real_qpd_a_vminvmax = [
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/conv/exp1/qpd-real/qpd-valid-best/real-qpd-test/375_epoch_564375_Mono-QPD/src/Dataset_A_1218/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp/real-qpd/qpd-valid-best/real-qpd-test/115_epoch_173075_Mono-QPD/vminvmax/A/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_skipconv-interp/real-qpd/qpd-valid-best/real-qpd-test/100_epoch_150500_Mono-QPD/vminvmax/A/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/QPDNet/result/eval/real-qpd/vminvmax/Dataset_A_1218/scale3/test_c/source/seq_1",
]

real_qpd_b_vminvmax = [
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/conv/exp1/qpd-real/qpd-valid-best/real-qpd-test/375_epoch_564375_Mono-QPD/src/Dataset_B_1218/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_interp/real-qpd/qpd-valid-best/real-qpd-test/115_epoch_173075_Mono-QPD/vminvmax/B/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple/result/eval/exp_skipconv-interp/real-qpd/qpd-valid-best/real-qpd-test/100_epoch_150500_Mono-QPD/vminvmax/B/scale3/test_c/source/seq_1",
    "/mnt/d/Mono+Dual/QPDNet/result/eval/real-qpd/vminvmax/Dataset_B_1218/scale3/test_c/source/seq_1",
]

if __name__ == "__main__":
    # ---- User input ----
    paths = real_qpd_b_vminvmax
    # ---- User input end ----

    ppt_img_dir = 'real_qpd_b_vminvmax_ppt_img'
    os.makedirs(ppt_img_dir, exist_ok=True)

    for i, pth in enumerate(paths):
        file_paths = glob(os.path.join(pth, '*'))
        for j, file in enumerate(file_paths):
            filename = os.path.basename(file)
            new_filename = f"{filename.replace('.png', '')}_{i:03d}.png"
            new_filepath = os.path.join(ppt_img_dir, new_filename)
            shutil.copy(file, new_filepath)
    