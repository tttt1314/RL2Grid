#!/usr/bin/env python3
"""Resumable job queue for RL2Grid runs on a (Spot) GCP VM.

Design
------
* Each job = one line of jobs.tsv x one seed.  State lives in  ~/queue/state/<job>_s<seed>/ .
* Runs are executed in *segments* of --segment-min minutes (RL2Grid's own --time-limit).
  At the end of every segment RL2Grid writes a checkpoint; the next segment resumes it with
  --resume-run-name.  => a Spot preemption loses at most one segment of work.
* A job is DONE when a segment exits well before its time limit (budget reached) or when the
  last "Eval at step N" in the log reaches its --total-timesteps.
* Concurrency is controlled by "slots" (jobs.tsv col 2) so CPU-hungry vanilla runs and
  straggler-bound heuristic runs can be mixed without oversubscribing the VM.
* Re-running this script is always safe: finished jobs are skipped, unfinished ones resume.

Usage (on the VM, from a checkout of tttt1314/RL2Grid):
    nohup python3 tools/queue/run_queue.py --jobs tools/queue/jobs_bus14.tsv \
        --seeds 0 1 2 --max-slots 4 --only PPO_ DQN_ > ~/queue/queue.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
RL2GRID = Path(__file__).resolve().parents[2]          # repo root (tools/queue/run_queue.py)
CKPT_DIR = RL2GRID / "checkpoint"
STATE = HOME / "queue" / "state"
PYTHON = HOME / "miniconda3" / "envs" / "rl2grid" / "bin" / "python"

# Launcher: turn SIGTERM (Spot preemption / shutdown) into KeyboardInterrupt so RL2Grid's
# `finally:` block still saves the checkpoint, then run main.py as __main__.
# Signals arriving within `window` seconds of the last one are ignored, so the queue AND systemd both
# signalling at shutdown cannot interrupt the checkpoint write. The window is finite (not "ignore forever"):
# a KeyboardInterrupt swallowed during env initialisation left a run that no SIGTERM could stop, so the
# queue re-sends SIGTERM every RESEND_SEC until the job exits.
RESEND_SEC = 120


def launcher(window: float) -> str:
    return (
        "import signal,sys,runpy,time\n"
        "_last=[None]\n"
        "def _stop(*a):\n"
        "    now=time.monotonic()\n"
        f"    if _last[0] is not None and now-_last[0]<{window}: return\n"
        "    _last[0]=now\n"
        "    raise KeyboardInterrupt\n"
        "signal.signal(signal.SIGTERM, _stop)\n"
        # a nohup'ed / backgrounded parent leaves SIGINT ignored; install the handler explicitly
        "signal.signal(signal.SIGINT, _stop)\n"
        "sys.argv=['main.py']+sys.argv[1:]\n"
        "runpy.run_path('main.py', run_name='__main__')\n"
    )


def ckpt_ok(path: Path) -> bool:
    """A torch.save archive must contain .../data.pkl and pass the zip CRC check.
    (zipfile.is_zipfile alone accepted an 825-byte truncated checkpoint that torch could not load.)"""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith("data.pkl") for n in z.namelist()) and z.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False

EVAL_RE = re.compile(r"Eval at step (\d+)")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def load_jobs(path: Path, seeds: list[int]) -> list[dict]:
    jobs = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, slots, extra = line.split("\t", 2)
        budget = int(re.search(r"--total-timesteps (\d+)", extra).group(1))
        for s in seeds:
            jobs.append(dict(id=f"{name}_s{s}", slots=int(slots), budget=budget, seed=s,
                             args=f"--env-id bus14 --difficulty 0 --seed {s} {extra}".split()))
    # seed-major: every config's seed 0 first, then seed 1, ... -> a complete first table appears early
    return sorted(jobs, key=lambda j: seeds.index(j["seed"]))


class Job:
    def __init__(self, spec: dict, segment_min: int, resend_sec: float = RESEND_SEC):
        self.__dict__.update(spec)
        self.dir = STATE / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.segment_min = segment_min
        self.proc: subprocess.Popen | None = None
        self.interrupted = False
        self.resend_sec = resend_sec
        self.last_term = 0.0
        self.t0 = 0.0

    # ---- persistent state -------------------------------------------------------------
    @property
    def done(self) -> bool:
        return (self.dir / "DONE").exists()

    @property
    def run_name(self) -> str:
        f = self.dir / "run_name"
        return f.read_text().strip() if f.exists() else ""

    def last_step(self) -> int:
        steps = [int(m) for f in sorted(self.dir.glob("seg_*.log"))
                 for m in EVAL_RE.findall(f.read_text(errors="ignore"))]
        return max(steps, default=0)

    # ---- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        seg = len(list(self.dir.glob("seg_*.log")))
        args = list(self.args) + ["--checkpoint", "True", "--time-limit", str(self.segment_min),
                                  "--exp-tag", self.id]
        if self.run_name:
            # RL2Grid deletes checkpoint/<run>.tar right after loading it, so keep a backup
            # in case this segment dies before its `finally:` re-saves it.
            ckpt, backup = CKPT_DIR / f"{self.run_name}.tar", self.dir / "backup.tar"
            if not ckpt_ok(ckpt):
                if ckpt_ok(backup):
                    log(f"WARN  {self.id}: checkpoint missing/corrupt, restoring last good backup")
                    shutil.copy2(backup, ckpt)
                else:
                    log(f"WARN  {self.id}: no valid checkpoint or backup, restarting this run from scratch")
                    for f in (ckpt, backup, self.dir / "run_name"): f.unlink(missing_ok=True)
            if ckpt_ok(ckpt):
                shutil.copy2(ckpt, backup)   # only ever back up a valid checkpoint
            if self.run_name:
                args += ["--resume-run-name", self.run_name]
        logf = open(self.dir / f"seg_{seg:03d}.log", "w")
        self.t0 = time.time()
        (self.dir / "seg_t0").write_text(str(self.t0))   # lets a restarted queue claim this segment's checkpoint
        self.proc = subprocess.Popen([str(PYTHON), "-u", "-c", launcher(0.75 * self.resend_sec), *args], cwd=RL2GRID,
                                     stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        log(f"START {self.id} seg={seg} resume={self.run_name or '-'} pid={self.proc.pid}")

    def ckpt_prefix(self) -> str:
        # run_name = {alg}_{env}_{T|R}_{seed}_{difficulty}_{H}_{I}_{C}_{time}_{rand}
        a = self.args
        alg = a[a.index("--alg") + 1]
        kind = "T" if "topology" in a else "R"
        seed = a[a.index("--seed") + 1]
        heur = "H" if "--use-heuristic" in a else ""
        idle = "I" if "reconnect" not in a else ""  # --heuristic-type defaults to idle -> "I" even for vanilla runs
        cons = {"1": "C1", "2": "C2"}.get(a[a.index("--constraints-type") + 1], "") if "--constraints-type" in a else ""
        return f"{alg}_bus14_{kind}_{seed}_0_{heur}_{idle}_{cons}_"

    def claim(self, claimed: set, since: float) -> None:
        for pattern in (f"{self.id}__*.tar", self.ckpt_prefix() + "*.tar"):
            cands = sorted((p for p in CKPT_DIR.glob(pattern)
                            if p.stem not in claimed and p.stat().st_mtime >= since),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                (self.dir / "run_name").write_text(cands[0].stem)
                claimed.add(cands[0].stem)
                log(f"CLAIM {self.id} -> {cands[0].stem}")
                return

    def recover(self, claimed: set) -> None:
        """Queue restarted after its first segment was interrupted before the checkpoint was claimed."""
        t0f = self.dir / "seg_t0"
        if not self.done and not self.run_name and t0f.exists():
            self.claim(claimed, float(t0f.read_text()) - 5)

    def poll(self, claimed: set[str]) -> bool:
        """Return True when the current segment has finished (state updated)."""
        if self.proc is None or self.proc.poll() is None:
            return False
        elapsed_min = (time.time() - self.t0) / 60
        if not self.run_name:  # first segment: find the checkpoint this process just wrote
            self.claim(claimed, self.t0)
        step = self.last_step()
        clean = self.proc.returncode == 0
        finished_early = elapsed_min < 0.9 * self.segment_min and clean
        # Only a clean exit counts: the log can show evals past the checkpoint when a segment was killed
        # (SIGKILL loses everything since the last save), so a budget-reaching log of a killed run is not DONE.
        # An interrupted run that did save resumes, ends its loop at once and exits cleanly -> DONE then.
        if (step >= self.budget * 0.98 and clean) or (finished_early and step > 0):
            (self.dir / "DONE").write_text(f"step={step}\n")
            log(f"DONE  {self.id} step={step}")
        else:
            log(f"SEG   {self.id} rc={self.proc.returncode} {elapsed_min:.0f}min step={step}/{self.budget}")
            if self.proc.returncode not in (0, None) and elapsed_min < 5 and not self.interrupted:
                (self.dir / "FAILED").write_text(f"rc={self.proc.returncode}\n")
                log(f"FAIL  {self.id} (crashed within 5 min, see {self.dir})")
        self.proc = None
        return True

    def interrupt(self) -> None:
        self.interrupted = True   # a shutdown is not a crash: never mark the job FAILED because of it
        if self.proc and self.proc.poll() is None:
            # SIGTERM to the main process only: the launcher turns it into KeyboardInterrupt, RL2Grid's finally:
            # saves the checkpoint and closes its workers. (SIGINT may be ignored when started via nohup/&.)
            os.kill(self.proc.pid, signal.SIGTERM)
            self.last_term = time.time()

    def resend(self) -> None:
        """Re-signal a job that is still alive RESEND_SEC after the last SIGTERM (its first one may have been
        swallowed during start-up). The launcher ignores signals inside its window, so an ongoing save is safe."""
        if self.proc and self.proc.poll() is None and time.time() - self.last_term >= self.resend_sec:
            log(f"RESEND {self.id}: still running, sending SIGTERM again")
            self.interrupt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(Path(__file__).with_name("jobs_bus14.tsv")))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-slots", type=int, default=4)
    ap.add_argument("--segment-min", type=int, default=360)  # most off-policy runs finish in one segment
    ap.add_argument("--resend-sec", type=float, default=RESEND_SEC,
                    help="re-send SIGTERM to a job still alive this long after the last one (when stopping)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these job-name prefixes")
    a = ap.parse_args()

    jobs = [Job(s, a.segment_min, a.resend_sec) for s in load_jobs(Path(a.jobs), a.seeds)]
    if a.only:
        jobs = [j for j in jobs if any(j.id.startswith(p) for p in a.only)]
    claimed = {j.run_name for j in jobs if j.run_name}
    for j in jobs:
        j.recover(claimed)
    running: list[Job] = []

    stopping = False

    def on_term(*_):
        nonlocal stopping
        stopping = True
        log("SIGTERM: asking running jobs to checkpoint")
        for j in running:
            j.interrupt()
    signal.signal(signal.SIGTERM, on_term)

    while True:
        running = [j for j in running if not j.poll(claimed)]
        if stopping:
            if not running:
                break
            for j in running:   # also covers a job started between the signal and its running.append()
                if not j.interrupted: j.interrupt()
                else: j.resend()
            time.sleep(2)
            continue
        pending = [j for j in jobs if not j.done and not (j.dir / "FAILED").exists() and j not in running]
        used = sum(j.slots for j in running)
        for j in pending:
            if stopping:   # SIGTERM may arrive during the stagger sleep below: never start a job after it
                break
            if used + j.slots <= a.max_slots:
                j.start()
                running.append(j)
                used += j.slots
                time.sleep(20)  # stagger env initialisation
        if stopping:
            continue
        if not running and not pending:
            log("ALL JOBS FINISHED")
            break
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
