"""Unattended training: runs the seeds one after another and keeps the game usable underneath them.

    uv run python -m gdrl supervise --run runs/ppo_sm --seeds 0 1 2 -- --advance-mode greedy --reload-slowdown 0.25

The arguments after `--` go to `python -m gdrl train`. For each seed the supervisor starts the trainer
(with `--resume` once the seed's run directory has a policy.pt) and watches it every few seconds:

- the screen is locked (`IOConsoleLocked`): the trainer is stopped at once with SIGTERM, because with a
  locked screen the game cannot come to the front and, on 2026-10-20, it stopped registering inputs (every
  episode from a checkpoint died on the same tick), which would add meaningless attempts. Training resumes
  when the screen is unlocked;
- the game's memory passes `--max-rss-gb` (it leaks about 0.4 MB per death): the trainer is stopped, the
  game restarted, and the seed resumed;
- the trainer exits without clearing (crash, failed level reload, bridge error, or 200 deaths in a row on
  the same tick): the game is restarted and the seed resumed, up to `--max-restarts` times per seed; a
  second same-tick stop in a row ends the seed, since then the checkpoint itself is unsurvivable.

A seed is done when its result.json says "cleared" or "budget". The supervisor holds a keep-awake
assertion that keeps the display on (`caffeinate -d -i`), so the screen does not lock on display sleep.
Game restarts go through Steam (`open steam://rungameid/322170`); a level only loads with the window in
front, which `osascript` can do while nobody else is using the Mac. If the game does not come up, the
supervisor runs scripts/refresh-steam-manifest.py once (the "Update Required" case in the README).
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("gdrl.supervise")

STEAM_URL = "steam://rungameid/322170"
GAME_PATTERN = "Geometry Dash$"


# -- pure helpers (tested offline) ---------------------------------------------------------------------------

def parse_console_locked(ioreg_text: str) -> bool:
    m = re.search(r'"IOConsoleLocked"\s*=\s*(Yes|No|true|false)', ioreg_text)
    return bool(m) and m.group(1) in ("Yes", "true")


def seed_finished(run_dir: Path) -> bool:
    path = run_dir / "result.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("stopped") in ("cleared", "budget")


def trainer_command(run_root: str, seed: int, train_args: list[str], run_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "gdrl", "train", "--run", run_root, "--seed", str(seed), *train_args]
    if (run_dir / "policy.pt").exists() and "--resume" not in cmd:
        cmd.append("--resume")
    return cmd


# -- system probes -----------------------------------------------------------------------------------------------

def console_locked() -> bool:
    out = subprocess.run(["ioreg", "-n", "Root", "-d1"], capture_output=True, text=True).stdout
    return parse_console_locked(out)


def game_pid() -> int | None:
    out = subprocess.run(["pgrep", "-f", GAME_PATTERN], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def game_rss_gb() -> float:
    pid = game_pid()
    if pid is None:
        return 0.0
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 / 1024 if out else 0.0


def bring_game_to_front() -> None:
    subprocess.run(["osascript", "-e", 'tell application "System Events" to set frontmost of '
                    '(first process whose name contains "Geometry Dash") to true'], capture_output=True)


# -- game control ------------------------------------------------------------------------------------------------

def bridge_ready(level_id: int, timeout: float = 5.0) -> bool:
    """True if the bridge answers and the level is loaded and started (loads it if needed)."""
    from gdbridge.client import BridgeError, GDBridge

    try:
        with GDBridge(timeout=timeout) as b:
            ping = b.ping()
            if ping.get("in_level"):
                s = b.state()
                if s.level_id == level_id and s.started:
                    return True
            bring_game_to_front()
            b.load_main(level_id, timeout=30.0)
            return True
    except (OSError, BridgeError) as e:
        log.info("bridge not ready: %s", str(e)[:160])
        return False


def kill_game() -> None:
    pid = game_pid()
    if pid is None:
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        if game_pid() is None:
            return
        time.sleep(1)
    if game_pid() is not None:
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)


def start_game(level_id: int, wait_s: float = 240.0) -> bool:
    subprocess.run(["open", STEAM_URL], capture_output=True)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(5)
        if game_pid() is None:
            continue
        bring_game_to_front()  # the loading screen and the level start only advance with the window in front
        if bridge_ready(level_id):
            return True
    return False


def restart_game(level_id: int) -> bool:
    for attempt in range(3):
        log.info("restarting the game (try %d)", attempt + 1)
        kill_game()
        if start_game(level_id):
            log.info("game up, level %d loaded (game memory %.1f GB)", level_id, game_rss_gb())
            return True
        if attempt == 0:
            log.info("game did not come up; refreshing the Steam manifest (README: Update Required)")
            kill_game()
            subprocess.run([sys.executable, "scripts/refresh-steam-manifest.py"], capture_output=True)
    return False


def wait_unlocked(poll_s: float = 10.0) -> None:
    if not console_locked():
        return
    log.info("screen locked: waiting for it to be unlocked")
    while console_locked():
        time.sleep(poll_s)
    log.info("screen unlocked")


# -- the loop -------------------------------------------------------------------------------------------------------

def supervise(run_root: str, seeds: list[int], train_args: list[str], *, level_id: int = 1, max_rss_gb: float = 8.0,
              max_restarts: int = 30, poll_s: float = 2.0) -> int:
    keep_awake = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
    try:
        for seed in seeds:
            run_dir = Path(run_root) / f"seed{seed}"
            restarts = degenerate = 0
            while not seed_finished(run_dir):
                if restarts > max_restarts:
                    log.info("seed %d: giving up after %d restarts", seed, restarts - 1)
                    return 1
                wait_unlocked()
                if not bridge_ready(level_id) and not restart_game(level_id):
                    log.info("the game does not come up; retrying in 60 s")
                    restarts += 1
                    time.sleep(60)
                    continue
                cmd = trainer_command(run_root, seed, train_args, run_dir)
                log.info("seed %d: %s", seed, " ".join(cmd[1:]))
                proc = subprocess.Popen(cmd)
                reason = None
                while proc.poll() is None:
                    time.sleep(poll_s)
                    if reason is None and console_locked():
                        reason = "screen locked"
                    elif reason is None and game_rss_gb() > max_rss_gb:
                        reason = f"game memory above {max_rss_gb:.0f} GB"
                    elif reason is None and game_pid() is None:
                        reason = "game exited"
                    if reason is not None:
                        log.info("seed %d: stopping the trainer: %s", seed, reason)
                        proc.send_signal(signal.SIGTERM)
                        try:
                            proc.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break
                if seed_finished(run_dir):
                    break
                if reason == "screen locked":
                    continue  # nothing wrong with the game; resume once unlocked
                stopped = json.loads((run_dir / "result.json").read_text()).get("stopped") if (run_dir / "result.json").exists() else None
                if stopped == "degenerate":
                    degenerate += 1
                    if degenerate >= 2:
                        log.info("seed %d: every episode died on the same tick again after a game restart; stopping the seed", seed)
                        break
                else:
                    degenerate = 0
                restarts += 1
                log.info("seed %d: trainer stopped (%s, exit %s); restart %d", seed, reason or "trainer exited",
                         proc.returncode, restarts)
                restart_game(level_id)
            log.info("seed %d done: %s", seed, (run_dir / "result.json").read_text().strip() if (run_dir / "result.json").exists() else "")
        return 0
    finally:
        keep_awake.terminate()
