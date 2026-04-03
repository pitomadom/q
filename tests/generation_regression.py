#!/usr/bin/env python3
import os
import pathlib
import pty
import re
import select
import subprocess
import sys
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
MERGES = ROOT / "q.merges"
CORPUS = ROOT / "q.txt"
EXPORT = ROOT / "weights" / "exported_weights.bin"
PROMPTS = [
    "warmth in the throat and pressure in the chest",
    "the forest remembers what the choir forgot",
]
CHAIN_RE = re.compile(r"^\[\s*\d+\]\s+[<>\*][ +]")


def make_tiny_corpus():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tf:
        tf.write(CORPUS.read_text(encoding="utf-8", errors="ignore")[:1800])
        return pathlib.Path(tf.name)


def compile_c():
    c_bin = pathlib.Path("/tmp/q_generation_regression")
    proc = subprocess.run(
        ["cc", "postgpt_q.c", "-O2", "-lm", "-o", str(c_bin)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"C compile failed\n{proc.stderr}")
    return c_bin


def first_chain_line(cmd, prompt, cwd, timeout=45):
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        text=False,
        close_fds=True,
    )
    os.close(slave)
    sent = False
    buf = ""
    started = time.time()
    try:
        while time.time() - started < timeout:
            r, _, _ = select.select([master], [], [], 0.5)
            if not r:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(master, 4096).decode("utf-8", errors="replace")
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            buf += chunk
            if not sent and "q>" in buf:
                os.write(master, (prompt + "\n").encode("utf-8"))
                sent = True
            for line in buf.splitlines():
                stripped = line.lstrip()
                if CHAIN_RE.match(stripped):
                    try:
                        os.write(master, b"quit\n")
                    except OSError:
                        pass
                    proc.terminate()
                    return line.strip(), buf
            if "bad magic" in buf:
                proc.terminate()
                raise AssertionError(f"{cmd[0]}: bad magic\n{buf[:4000]}")
        proc.terminate()
        raise AssertionError(f"timeout waiting for first chain line\n{buf[:4000]}")
    finally:
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        os.close(master)


def check_engine(label, cmd, prompt, cwd, timeout=45):
    print(f"[run] {label} :: {prompt[:40]}", flush=True)
    line, full = first_chain_line(cmd, prompt, cwd, timeout=timeout)
    if "[...]" in line:
        raise AssertionError(f"{label}: first chain step collapsed\n{line}\n\n{full[:4000]}")
    if "{interf}" not in line and "{wormhole}" not in line and len(line) < 20:
        raise AssertionError(f"{label}: first chain line too weak\n{line}\n\n{full[:4000]}")
    print(f"[ok] {label}", flush=True)
    return line


def main():
    deep = "--deep" in sys.argv[1:]
    tiny = make_tiny_corpus()
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="q_regression_"))
    try:
        c_bin = compile_c()
        (workspace / "q.merges").write_bytes(MERGES.read_bytes())
        (workspace / "q_tiny.txt").write_bytes(tiny.read_bytes())
        if EXPORT.exists():
            (workspace / "exported_weights.bin").write_bytes(EXPORT.read_bytes())
        results = []
        for prompt in PROMPTS:
            results.append(("c-meta", prompt, check_engine("c-meta", [str(c_bin), "q.merges", "q_tiny.txt"], prompt, workspace, timeout=20)))
            if EXPORT.exists():
                results.append(("c-trained", prompt, check_engine("c-trained", [str(c_bin), "exported_weights.bin", "q.merges", "q_tiny.txt"], prompt, workspace, timeout=20)))
            if deep:
                results.append(("py-meta", prompt, check_engine("py-meta", [sys.executable, str(ROOT / "postgpt_q.py"), "q.merges", "q_tiny.txt"], prompt, workspace, timeout=75)))
                if EXPORT.exists():
                    results.append(("py-trained", prompt, check_engine("py-trained", [sys.executable, str(ROOT / "postgpt_q.py"), "exported_weights.bin", "q.merges", "q_tiny.txt"], prompt, workspace, timeout=75)))

        print("generation regression summary:")
        for engine, prompt, line in results:
            print(f"- {engine} | {prompt[:32]}... | {line}")
        if not deep:
            print("note: Python engines are available via --deep for slower parity runs")
        return 0
    finally:
        try:
            tiny.unlink()
        except OSError:
            pass
        try:
            for item in workspace.iterdir():
                item.unlink()
            workspace.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
