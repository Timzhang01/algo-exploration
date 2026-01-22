import argparse
import time
from multiprocessing import Process, Pipe, Queue


def worker(idx, left_conn, right_conn, steps, payload_size, result_q):
    payload = b"x" * payload_size
    t0 = time.time()
    for _ in range(steps):
        if idx % 2 == 0:
            right_conn.send_bytes(payload)
            left_conn.recv_bytes()
        else:
            left_conn.recv_bytes()
            right_conn.send_bytes(payload)
    dt = time.time() - t0
    result_q.put((idx, dt))


def build_ring(n):
    links = []
    for i in range(n):
        a, b = Pipe(duplex=True)
        links.append((a, b))
    left = [None] * n
    right = [None] * n
    for i in range(n):
        right[i] = links[i][0]
        left[(i + 1) % n] = links[i][1]
    return left, right


def run(n_workers, payload_bytes):
    steps = max(1, n_workers - 1)
    left, right = build_ring(n_workers)
    q = Queue()
    procs = []
    for i in range(n_workers):
        p = Process(target=worker, args=(i, left[i], right[i], steps, payload_bytes, q))
        p.start()
        procs.append(p)
    times = []
    for _ in range(n_workers):
        times.append(q.get())
    for p in procs:
        p.join()
    times.sort()
    max_time = max(t for _, t in times)
    total_bytes = steps * payload_bytes * n_workers
    throughput = total_bytes / max_time / (1024 * 1024)
    print(f"workers={n_workers} steps={steps} payload_bytes={payload_bytes}")
    print(f"max_worker_time_s={max_time:.4f}")
    print(f"aggregate_bytes={total_bytes}")
    print(f"aggregate_throughput_MBps={throughput:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    args = parser.parse_args()
    run(args.workers, args.payload_bytes)
