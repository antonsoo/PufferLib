"""Exercise native drone training through the CLI (requires a CUDA GPU).

    bash build.sh drone build/drone --float
    python tests/test_train_minibatches.py build/drone
"""

from array import array
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ERROR = "train.replay_ratio must schedule at least one minibatch per epoch"


def run(binary, output, ratio, minibatch, graphs):
    result = subprocess.run(
        [
            str(binary), "train", "--headless",
            f"--base.checkpoint_dir={output / 'checkpoints'}",
            f"--base.log_dir={output / 'logs'}",
            "--base.run_id=test", "--base.checkpoint_interval=1",
            "--base.eval_episodes=16", f"--base.cudagraphs={graphs}",
            "--vec.total_agents=256", "--vec.num_buffers=4",
            "--vec.num_threads=4", "--policy.hidden_size=64",
            "--policy.num_layers=2", "--train.horizon=64",
            "--train.total_timesteps=131072",
            f"--train.replay_ratio={ratio}",
            f"--train.minibatch_size={minibatch}",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    return result, sorted((output / "checkpoints").glob("drone/test/*.bin"))


def main():
    binary = Path(sys.argv[1]).resolve()
    assert binary.is_file(), binary
    for graphs in (-1, 1):
        for ratio, minibatch in ((0.5, 16384), (0.999, 16384), (0, 4096), (-1, 4096)):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                result, checkpoints = run(binary, output, ratio, minibatch, graphs)
                assert result.returncode != 0, "training accepted zero minibatches"
                assert ERROR in result.stderr, result.stderr
                assert not (output / "checkpoints").exists(), checkpoints
            print(f"PASS rejected replay_ratio={ratio}, minibatch={minibatch}, graphs={graphs}")

        # Fractional replay is valid when the rollout contains enough minibatches.
        for ratio, minibatch in ((0.25, 4096), (0.5, 8192), (1, 16384)):
            with tempfile.TemporaryDirectory() as tmp:
                result, checkpoints = run(binary, Path(tmp), ratio, minibatch, graphs)
                assert result.returncode == 0, result.stdout + result.stderr
                assert len(checkpoints) == 8, checkpoints
                first = array("f", checkpoints[0].read_bytes())
                last = array("f", checkpoints[-1].read_bytes())
                assert len(first) == len(last)
                assert all(math.isfinite(x) for x in last)
                # Check encoder weights, excluding any allocator padding.
                assert max(abs(a - b) for a, b in zip(first[:512], last[:512])) > 1e-6
            print(f"PASS updated weights: replay_ratio={ratio}, minibatch={minibatch}, graphs={graphs}")


if __name__ == "__main__":
    main()
