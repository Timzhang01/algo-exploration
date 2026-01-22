import argparse
import socket
import threading
import time


def server(port, rtts, bulk_size):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    conn, _ = s.accept()
    for _ in range(rtts):
        conn.recv(1024)
        conn.sendall(b"a")
    received = 0
    target = bulk_size
    while received < target:
        chunk = conn.recv(min(1 << 20, target - received))
        if not chunk:
            break
        received += len(chunk)
    conn.close()
    s.close()


def client(port, rtts, bulk_size):
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect(("127.0.0.1", port))
    t0 = time.time()
    for _ in range(rtts):
        t1 = time.time()
        c.sendall(b"x" * 1024)
        c.recv(1)
        t2 = time.time()
        _ = t2 - t1
    rtt_time = (time.time() - t0) / rtts
    payload = b"y" * bulk_size
    t3 = time.time()
    sent = 0
    while sent < bulk_size:
        n = c.send(payload[sent: sent + (1 << 20)])
        if n <= 0:
            break
        sent += n
    t4 = time.time()
    c.close()
    throughput = bulk_size / (t4 - t3) / (1024 * 1024)
    print(f"avg_rtt_ms={rtt_time * 1000:.3f}")
    print(f"bulk_size_bytes={bulk_size} throughput_MBps={throughput:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--rtts", type=int, default=1000)
    parser.add_argument("--bulk-size", type=int, default=1 << 24)
    args = parser.parse_args()
    th = threading.Thread(target=server, args=(args.port, args.rtts, args.bulk_size))
    th.daemon = True
    th.start()
    time.sleep(0.2)
    client(args.port, args.rtts, args.bulk_size)
