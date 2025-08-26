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
    # latest.pth 유무에 상관없이 프로그램이 알아서 처리하므로 항상 넣는다.
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
    res = subprocess.run(f"qsub {script_path}", shell=True, capture_output=True, text=True)
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

    # ✅ scripts/<run_setting_name>/ 구조로 정리
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

    # ===== train 모드(단일 잡 내부에 3 프로세스 병렬) =====
    train_header = pick_header_env_for_train()
    train_jobname = f"{args.run_setting_name}_train_pack"
    base = f"pack_{args.run_setting_name}_{ts}"
    script_path = scripts_dir / f"{base}.sh"
    out_log = scripts_dir / f"{base}_out.log"
    err_log = scripts_dir / f"{base}_err.log"

    train_cmd = build_train_cmd(args.run_setting_name, checkpoints_dir)
    eval_cmd = build_eval_cmd(args.run_setting_name, "latest", args.eval_datasets)  # watcher는 항상 latest로 평가

    # 평가 서브잡 헤더(HereDoc로 제출할 때 박아넣는다)
    eval_header_text = render_header(
        pick_header_env_for_eval(),
        f"{args.run_setting_name}_eval",
        scripts_dir / f"{base}_eval_out.log",
        scripts_dir / f"{base}_eval_err.log",
    ).rstrip()

    # 재제출 시간(23h55m)
    # limit_sec = 23 * 3600 + 55 * 60
    limit_sec = 4 * 60

    # 단일 잡 스크립트(3개 프로세스 병렬: TRAIN(포그라운드), WATCHER(백), RESTARTER(백))
    script = f"""#!/bin/bash
{render_header(train_header, train_jobname, out_log, err_log)}
cd {exec_path}

SELF="{script_path.resolve()}"
CKPT_DIR="{checkpoints_dir.resolve()}"
POLL=20
SETTLE=10
LIMIT={limit_sec}

echo "[pack] start single job: train + watcher + restarter"

# ---- WATCHER: 최신 체크포인트 감시 후 평가 서브잡 제출 ----
watcher_loop() {{
  echo "[watcher] watching $CKPT_DIR every $POLL s"
  LAST=""
  while true; do
    NEWEST=$(ls -1t "$CKPT_DIR"/*.pth 2>/dev/null | head -n1)
    if [ -n "$NEWEST" ] && [ "$NEWEST" != "$LAST" ]; then
      echo "[watcher] detected new ckpt: $NEWEST"
      sleep $SETTLE
      LAST="$NEWEST"
      qsub <<'EOF'
#!/bin/bash
{eval_header_text}
cd {exec_path}
{eval_cmd}
EOF
    fi
    sleep $POLL
  done
}}

# ---- RESTARTER: final.pth 있으면 종료, 아니면 시간 도달 시 자기 자신 재제출 ----
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
      echo "[restarter] time reached; re-submit SELF: $SELF"
      qsub "$SELF"
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

    write_and_submit(script_path, script, do_submit)


if __name__ == "__main__":
    main()
