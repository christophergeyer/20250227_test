import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
from boto3.s3.transfer import TransferConfig


@dataclass
class Stats:
    label: str
    bytes_total: int
    seconds: float

    @property
    def mib(self) -> float:
        return self.bytes_total / (1024 * 1024)

    @property
    def mib_per_s(self) -> float:
        return self.mib / self.seconds if self.seconds > 0 else float("inf")


def now() -> float:
    return time.perf_counter()


def gen_object_keys(prefix: str, n: int) -> list[str]:
    # deterministic-ish names for easier cleanup and comparisons
    return [f"{prefix}/obj_{i:05d}.bin" for i in range(n)]


def write_local_file(path: str, size_bytes: int, chunk_bytes: int) -> None:
    # Avoid /dev/urandom cost; use PRNG-filled blocks.
    # High variability is OK; we just need relative comparisons.
    rnd = random.Random(0xC0FFEE)
    with open(path, "wb", buffering=0) as f:
        remaining = size_bytes
        while remaining > 0:
            n = min(chunk_bytes, remaining)
            # build a block quickly; not cryptographically random
            block = bytes(rnd.getrandbits(8) for _ in range(n))
            f.write(block)
            remaining -= n


def s3_upload_many(s3, bucket: str, keys: list[str], local_path: str, part_size: int, max_conc: int) -> Stats:
    cfg = TransferConfig(
        multipart_threshold=part_size,  # force multipart at/above part_size
        multipart_chunksize=part_size,
        max_concurrency=max_conc,
        use_threads=True,
    )
    file_size = os.path.getsize(local_path)

    def _upload_one(key: str) -> int:
        s3.upload_file(local_path, bucket, key, Config=cfg)
        return file_size

    t0 = now()
    total = 0
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futures = [pool.submit(_upload_one, k) for k in keys]
        for f in as_completed(futures):
            total += f.result()
    t1 = now()
    return Stats("S3 PUT (upload_file)", total, t1 - t0)


def s3_download_many(s3, bucket: str, keys: list[str], read_chunk: int, max_conc: int) -> Stats:
    def _download_one(key: str) -> int:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]
        n = 0
        while True:
            data = body.read(read_chunk)
            if not data:
                break
            n += len(data)
        return n

    t0 = now()
    total = 0
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futures = [pool.submit(_download_one, k) for k in keys]
        for f in as_completed(futures):
            total += f.result()
    t1 = now()
    return Stats("S3 GET (stream read)", total, t1 - t0)


def s3_range_read_many(s3, bucket: str, keys: list[str], ranges_per_obj: int, range_size: int, max_conc: int) -> Stats:
    # Mimic shard-style random access: many small ranged GETs.
    # This can show bigger differences once endpoint is in place.

    def _range_read_one(key: str) -> int:
        head = s3.head_object(Bucket=bucket, Key=key)
        size = head["ContentLength"]
        n = 0
        for _ in range(ranges_per_obj):
            if size <= range_size:
                start = 0
            else:
                start = random.randint(0, size - range_size)
            end = start + range_size - 1
            resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
            data = resp["Body"].read()
            n += len(data)
        return n

    t0 = now()
    total = 0
    with ThreadPoolExecutor(max_workers=max_conc) as pool:
        futures = [pool.submit(_range_read_one, k) for k in keys]
        for f in as_completed(futures):
            total += f.result()
    t1 = now()
    return Stats(f"S3 GET Range ({ranges_per_obj}x{range_size}B per obj)", total, t1 - t0)


def s3_delete_many(s3_client, bucket: str, keys: list[str]) -> None:
    # Batch delete in chunks of 1000
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i+1000]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )


def main():
    ap = argparse.ArgumentParser(description="Quick S3 micro-benchmark (~1 minute)")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="bench/microbench")
    ap.add_argument("--keys", type=int, default=16, help="number of objects")
    ap.add_argument("--obj-mib", type=int, default=8, help="object size in MiB")
    ap.add_argument("--write-chunk-kib", type=int, default=1024, help="local generation chunk KiB")
    ap.add_argument("--part-mib", type=int, default=8, help="multipart part size MiB (also threshold)")
    ap.add_argument("--read-chunk-kib", type=int, default=1024, help="streaming read chunk KiB")
    ap.add_argument("--max-concurrency", type=int, default=4, help="multipart concurrency")
    ap.add_argument("--range-reads-per-obj", type=int, default=0, help="if >0, do random range reads")
    ap.add_argument("--range-kib", type=int, default=256, help="range size KiB")
    ap.add_argument("--keep", action="store_true", help="keep uploaded objects (no cleanup)")
    args = ap.parse_args()

    bucket = args.bucket
    prefix = args.prefix.rstrip("/")

    keys = gen_object_keys(prefix, args.keys)

    obj_bytes = args.obj_mib * 1024 * 1024
    write_chunk = args.write_chunk_kib * 1024
    part_size = args.part_mib * 1024 * 1024
    read_chunk = args.read_chunk_kib * 1024
    range_size = args.range_kib * 1024

    # boto3 clients are thread-safe; one client is sufficient for all operations.
    s3 = boto3.client("s3")

    # Local file
    local_path = f"/tmp/s3_microbench_{args.obj_mib}MiB.bin"
    print(f"Generating local file: {local_path} size={args.obj_mib}MiB chunk={args.write_chunk_kib}KiB")
    t0 = now()
    write_local_file(local_path, obj_bytes, write_chunk)
    t1 = now()
    gen_secs = t1 - t0
    print(f"Local gen: {obj_bytes/(1024*1024):.1f} MiB in {gen_secs:.3f}s ({(obj_bytes/(1024*1024))/gen_secs:.1f} MiB/s)")

    # Upload
    print(f"\nUploading: keys={args.keys}, part={args.part_mib}MiB, concurrency={args.max_concurrency}")
    put_stats = s3_upload_many(
        s3,
        bucket=bucket,
        keys=keys,
        local_path=local_path,
        part_size=part_size,
        max_conc=args.max_concurrency,
    )
    print(f"{put_stats.label}: {put_stats.mib:.1f} MiB in {put_stats.seconds:.3f}s = {put_stats.mib_per_s:.1f} MiB/s")

    # Download streaming
    print(f"\nDownloading (stream): read_chunk={args.read_chunk_kib}KiB")
    max_conc = args.max_concurrency
    get_stats = s3_download_many(s3, bucket=bucket, keys=keys, read_chunk=read_chunk, max_conc=max_conc)
    print(f"{get_stats.label}: {get_stats.mib:.1f} MiB in {get_stats.seconds:.3f}s = {get_stats.mib_per_s:.1f} MiB/s")

    # Range reads (optional)
    if args.range_reads_per_obj > 0:
        print(f"\nRange reads: per_obj={args.range_reads_per_obj}, range={args.range_kib}KiB")
        rr_stats = s3_range_read_many(
            s3, bucket=bucket, keys=keys,
            ranges_per_obj=args.range_reads_per_obj,
            range_size=range_size,
            max_conc=max_conc,
        )
        print(f"{rr_stats.label}: {rr_stats.mib:.1f} MiB in {rr_stats.seconds:.3f}s = {rr_stats.mib_per_s:.1f} MiB/s")

    # Cleanup
    if args.keep:
        print("\nKeeping objects (no cleanup).")
    else:
        print("\nCleaning up uploaded objects...")
        s3_delete_many(s3, bucket=bucket, keys=keys)
        print("Cleanup done.")

    try:
        os.remove(local_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
