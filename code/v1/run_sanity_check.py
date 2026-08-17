from __future__ import annotations
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Quick 1/5/10-item sanity check before full calibration.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-dir", default="runs/sanity_v03")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "run_calibration.py",
        "--model", args.model,
        "--device", args.device,
        "--sizes", "1,5,10",
        "--trials", "1",
        "--prompt-style", "base",
        "--out-dir", args.out_dir,
    ]
    subprocess.run(cmd, check=True)
    print("\nSanity run complete. Inspect the three response files, then run:")
    print(f"python summarize_calibration.py --dir {args.out_dir} --required-passes 1")


if __name__ == "__main__":
    main()
