import argparse
import time


def simulate_step(compute_ms, comm_ms, overlap):
    c = compute_ms / 1000.0
    m = comm_ms / 1000.0
    if overlap:
        t0 = time.time()
        time.sleep(max(c, m))
        return time.time() - t0, c / max(c, m)
    else:
        t0 = time.time()
        time.sleep(c + m)
        return time.time() - t0, c / (c + m)


def run(steps, compute_ms, comm_ms, overlap, accumulate):
    total_time = 0.0
    total_compute = 0.0
    total_comm = 0.0
    mfu_samples = []
    acc = max(1, accumulate)
    for i in range(steps):
        this_comm = comm_ms if (i + 1) % acc == 0 else 0
        step_time, mfu = simulate_step(compute_ms, this_comm, overlap)
        total_time += step_time
        total_compute += compute_ms / 1000.0
        total_comm += this_comm / 1000.0
        mfu_samples.append(mfu)
    avg_mfu = sum(mfu_samples) / len(mfu_samples)
    throughput = steps / total_time
    print(f"steps={steps}")
    print(f"compute_ms={compute_ms} comm_ms={comm_ms} overlap={overlap} accumulate={accumulate}")
    print(f"total_time_s={total_time:.4f}")
    print(f"total_compute_s={total_compute:.4f} total_comm_s={total_comm:.4f}")
    print(f"avg_mfu={avg_mfu:.4f}")
    print(f"steps_per_second={throughput:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--compute-ms", type=float, default=20.0)
    parser.add_argument("--comm-ms", type=float, default=10.0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--accumulate", type=int, default=1)
    args = parser.parse_args()
    run(args.steps, args.compute_ms, args.comm_ms, args.overlap, args.accumulate)
