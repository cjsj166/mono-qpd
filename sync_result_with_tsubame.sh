#!/bin/bash
cd /mnt/d/Mono+Dual/mono-qpd/mono-qpd-simple
watch -d -n 600 "rsync -avz --exclude='*.pth' -e ssh ua04628@login2.t4.gsic.titech.ac.jp:/gs/bs/tga-lab_otm/dlee/mono-qpd/result/ tsubame_result/"