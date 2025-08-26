#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from time import sleep
from presets import get_run_setting
from presets.header_env_settings import FMDPTrain, FMDPValid


def parse_args():
    p = argparse.ArgumentParser("TSUBAME single-job (train + watcher + restarter)")
    p.add_argument("--type", choices=["train", "eval"], required=True)
    p.add_argument("--run_setting_name", required=True)
    p.add_argument("--after", type=int, default=0, help="N초 대기 후 제출")
    p.add_argument("--run_script", default="True", choices=["True", "False"])
    # eval 모드 전용
    p.add_argument("--ckpt_epoch", default="latest")
    p.add_argument("--eval_datasets", nargs="+", default=["DPD_Disp", "QPD-Valid"])
    return p.parse_args()


def pick_header_env_for_train():
    h = FMDPTrain()
    return {
        "node_type": h.node_type,
        "running_time": h.running_time,
        "env_name": h.env_name,
        "cuda_version": h.cuda_version,
        "cudnn_version": h.cudnn_version,
    }


def pick_header_env_for_eval():
    h = FMDPValid()
    return {
        "node_type": h.node_type,
        "running_time": h.running_time,
        "env_name": h.env_name,
        "cuda_version": h.cuda_version,
        "cudnn_version": h.cudnn_version,
    }


def render_header(header_env: dict, job_name: str, stdout_log: Path, stderr_log: Path) -> str:
    # 이 헤더는 스크립트 파일 맨 위에만 들어가도록 사용한다.
    return f"""#$ -cwd
#$ -o {stdout_log}
#$ -e {stderr_log}
#$ -N {job_name}
#$ -l {header_env["node_type"]}=1
#$ -l h_rt={header_env["running_time"]}
#$ -V
. /etc/profile.d/modules.sh
module load {header_env["cuda_version"]} {header_env["cudnn_version"]}
source ~/.bashrc
conda activate {header_env["env_name"]}
"""


def build_train_cmd(run_setting_name: str, checkpoints_dir: Path) -> str:
    # latest.pth 유무는 프로그램 내부에서 처리하므로 항상 넣는다.
    return f"python train_mono_qpd.py --exp_name {run_setting_name} --restore_ckpt latest"


def build_eval_cmd(run_setting_name: str, ckpt_epoch: str, eval_datasets: list[str]) -> str:
    ds = " ".join(eval_datasets)
    return (
        f"python evaluate_mono_qpd.py "
        f"--exp_name {run_setting_name} "
        f"--ckpt_epoch {ckpt_epoch} "
        f"--eval_datasets {ds}"
    )


def write_and_submit(script_path: Path, text: str, do_submit: bool) -> str:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(text)
    script_path.chmod(0o755)
    print(str(script_path.resolve()))
    if not do_submit:
        return ""
    # 그룹은 qsub 옵션에서 지정
    res = subprocess.run(f"qsub -g tga-lab_okmn {script_path}", shell=True, capture_output=True, text=True)
    out = res.stdout.strip()
    err = res.stderr.strip()
    if out:
        print(out)
    if err:
        print(err)
    return out


def main():
    args = parse_args()
    if args.after > 0:
        sleep(args.after)

    # 런세팅(여기서 save_path 사용)
    run_setting = get_run_setting(args.run_setting_name)
    save_path = Path(run_setting.save_path)
    checkpoints_dir = save_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    exec_path = Path.cwd()

    # scripts/<run_setting_name>/ 구조
    root_scripts_dir = exec_path / "scripts"
    scripts_dir = root_scripts_dir / args.run_setting_name
    scripts_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    do_submit = (args.run_script == "True")

    # ===== eval 모드(단발) =====
    if args.type == "eval":
        eval_header = pick_header_env_for_eval()
        jobname = f"{args.run_setting_name}_eval_once"
        base = f"eval_once_{args.run_setting_name}_{ts}"
        script_path = scripts_dir / f"{base}.sh"
        out_log = scripts_dir / f"{base}_out.log"
        err_log = scripts_dir / f"{base}_err.log"
        eval_cmd = build_eval_cmd(args.run_setting_name, args.ckpt_epoch, args.eval_datasets)

        script = f"""#!/bin/bash
{render_header(eval_header, jobname, out_log, err_log)}
cd {exec_path}
{eval_cmd}
"""
        write_and_submit(script_path, script, do_submit)
        return

    # ===== train 모드 =====
    train_header = pick_header_env_for_train()
    train_jobname = f"{args.run_setting_name}_train_pack"
    base = f"pack_{args.run_setting_name}_{ts}"
    pack_script_path = scripts_dir / f"{base}.sh"
    out_log = scripts_dir / f"{base}_out.log"
    err_log = scripts_dir / f"{base}_err.log"

    train_cmd = build_train_cmd(args.run_setting_name, checkpoints_dir)
    eval_cmd = build_eval_cmd(args.run_setting_name, "latest", args.eval_datasets)  # watcher는 항상 latest로 평가

    # ✅ (중요) 평가 전용 스크립트를 별도 파일로 저장
    eval_header = pick_header_env_for_eval()
    eval_jobname = f"{args.run_setting_name}_eval"
    eval_script_path = scripts_dir / f"eval_runner_{args.run_setting_name}.sh"  # 고정 파일명
    eval_out_log = scripts_dir / f"eval_runner_{args.run_setting_name}_out.log"
    eval_err_log = scripts_dir / f"eval_runner_{args.run_setting_name}_err.log"

    eval_script_text = f"""#!/bin/bash
{render_header(eval_header, eval_jobname, eval_out_log, eval_err_log)}
cd {exec_path}
{eval_cmd}
"""
    eval_script_path.write_text(eval_script_text)
    eval_script_path.chmod(0o755)

    # 재제출 시간 (운영: 23h55m). 지금은 테스트 값.
    # limit_sec = 23 * 3600 + 55 * 60
    limit_sec = 9 * 60

    # ---- pack 스크립트(학습 + watcher + restarter) 본문 ----
    pack_script_text = f"""#!/bin/bash
{render_header(train_header, train_jobname, out_log, err_log)}
cd {exec_path}

echo "[info] JOB_ID=$JOB_ID JOB_NAME=$JOB_NAME"

SELF="{pack_script_path.resolve()}"
CKPT_DIR="{checkpoints_dir.resolve()}"
EVAL_SCRIPT="{eval_script_path.resolve()}"
POLL="${{POLL:-60}}"
SETTLE="${{SETTLE:-10}}"
LIMIT={limit_sec}

echo "[pack] start single job: train + watcher + restarter"

# ---- WATCHER: latest.pth 갱신 시에만 평가 서브잡 제출 ----
watcher_loop() {{
  echo "[watcher] watching latest: $CKPT_DIR/latest.pth (poll=${{POLL}}s, settle=${{SETTLE}}s)"

  # 파일 쓰기 안정화 체크
  stable_size() {{
    local f="$1"
    local s1 s2
    s1=$(stat -c %s "$f" 2>/dev/null || echo 0)
    sleep "$SETTLE"
    s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [[ "$s2" -gt 0 && "$s1" -eq "$s2" ]]
  }}

  LAST_SIG=""

  while true; do
    if [[ -L "$CKPT_DIR/latest.pth" ]]; then
      SIG=$(readlink -f "$CKPT_DIR/latest.pth" 2>/dev/null || echo "")
    else
      SIG=""
    fi

    if [[ -n "$SIG" && "$SIG" != "$LAST_SIG" ]]; then
      TARGET="$SIG"
      if [[ -f "$TARGET" ]] && stable_size "$TARGET"; then
        echo "[watcher] latest updated → $TARGET"
        qsub -g tga-lab_okmn "$EVAL_SCRIPT"
        LAST_SIG="$SIG"
      fi
    fi

    sleep "$POLL"
  done
}}

# ---- RESTARTER: final.pth 발견 시 종료, 아니면 시간 도달 시 자기 자신을 hold 의존으로 재제출 ----
restarter_loop() {{
  echo "[restarter] limit=$LIMIT s; early exit if final.pth appears"
  local start=$(date +%s)
  while true; do
    if [ -f "$CKPT_DIR/final.pth" ]; then
      echo "[restarter] final.pth detected; stop restarter"
      return 0
    fi
    now=$(date +%s); elapsed=$((now-start))
    if [ $elapsed -ge $LIMIT ]; then
      echo "[restarter] time reached; re-submit SELF with hold on current JOB_ID=$JOB_ID: $SELF"
      qsub -g tga-lab_okmn -hold_jid "$JOB_ID" "$SELF"
      return 0
    fi
    sleep 30
  done
}}

# ---- TRAIN(포그라운드) + WATCHER/RESTARTER(백그라운드) ----
watcher_loop & WATCHER_PID=$!
restarter_loop & RESTARTER_PID=$!

echo "[train] run: {train_cmd}"
{train_cmd}
TRAIN_RC=$?

# 학습 종료 시 보조 루프 정리
kill $WATCHER_PID $RESTARTER_PID >/dev/null 2>&1 || true
wait $WATCHER_PID $RESTARTER_PID 2>/dev/null || true

echo "[pack] train finished rc=$TRAIN_RC"
exit $TRAIN_RC
"""

    # pack 스크립트 쓰고 제출
    write_and_submit(pack_script_path, pack_script_text, do_submit)


if __name__ == "__main__":
    main()
