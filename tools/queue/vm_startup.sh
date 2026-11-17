#!/bin/bash
# GCE startup-script (runs as root on every boot).
# Restarts the RL2Grid job queue after a Spot preemption + `gcloud compute instances start`.
# Per-VM settings live in /home/<user>/queue/queue_args (e.g. "--only PPO_ DQN_ TD3_").
# Finished jobs are skipped and unfinished ones resume from their last checkpoint.
U=tttt1123
H=/home/$U
[ -f "$H/queue/queue_args" ] || exit 0
sudo -u "$U" bash -c "
  cd $H/RL2Grid-fork || exit 0
  pgrep -f 'tools/queue/run_queu[e].py' >/dev/null && exit 0
  echo \"\$(date '+%F %T') BOOT: (re)starting queue\" >> $H/queue/queue.log
  nohup python3 tools/queue/run_queue.py --jobs tools/queue/jobs_bus14.tsv --seeds 0 1 2 --max-slots 4 \
      \$(cat $H/queue/queue_args) >> $H/queue/queue.log 2>&1 < /dev/null &
"
