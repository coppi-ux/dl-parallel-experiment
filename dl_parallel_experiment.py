"""
Multithreading vs Multiprocessing in a Deep Learning pipeline
-------------------------------------------------------------
Measures how sequential, multithreaded and multiprocess data loading affect
training time, throughput and CPU / RAM / GPU usage.

How to run (in the VS Code terminal, with your virtual environment active):

    python dl_parallel_experiment.py --data-dir data/seg_train/seg_train --quick   # tiny test run
    python dl_parallel_experiment.py --data-dir data/seg_train/seg_train           # full experiment

--data-dir must be a folder with one sub-folder per class, e.g.
    data/seg_train/seg_train/buildings/*.jpg
    data/seg_train/seg_train/forest/*.jpg
Works for Intel Image Classification (seg_train/seg_train) and Cats vs Dogs (PetImages).

Outputs (saved next to this script): hardware_specs.txt, raw_runs.csv, results.csv,
time_vs_workers.png, throughput_vs_workers.png, gpu_vs_workers.png
"""

import argparse
import gc
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

# ------------------------------------------------------------------ CONFIG
BATCH_SIZE = 64
EPOCHS = 3
IMG_SIZE = 128
MAX_SAMPLES = 10000            # limit dataset size so each run stays short (None = use all)
THREAD_COUNTS = [2, 4, 8]
PROCESS_COUNTS = [1, 2, 4, 8]  # adjust to your CPU
REPEATS = 2                    # runs per configuration, results are averaged
SEED = 42
# --------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_HANDLE = None


# ------------------------------------------------------------ hardware specs
def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_specs(dataset_size, batch_size, epochs):
    return {
        "CPU": cpu_model(),
        "CPU cores (physical)": psutil.cpu_count(logical=False),
        "CPU threads (logical)": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / 1e9, 1),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)",
        "Operating System": platform.platform(),
        "Python version": platform.python_version(),
        "Framework": f"PyTorch {torch.__version__} (torchvision {torchvision.__version__}), CUDA {torch.version.cuda}",
        "Dataset size": dataset_size,
        "Batch size": batch_size,
        "Epochs": epochs,
    }


# ------------------------------------------------------------------- dataset
def safe_loader(path):
    """Some files (e.g. in Cats vs Dogs) are corrupted; return a blank image instead of crashing."""
    try:
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")
    except Exception:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE))


def build_dataset(data_dir, max_samples):
    # Deliberately non-trivial preprocessing so the CPU has real work to do
    tf = T.Compose([
        T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = ImageFolder(data_dir, transform=tf, loader=safe_loader)
    if max_samples and len(full) > max_samples:
        idx = np.random.RandomState(SEED).permutation(len(full))[:max_samples]
        return Subset(full, idx), len(full.classes), full.classes
    return full, len(full.classes), full.classes


# --------------------------------------------------------------- data loaders
class ThreadedLoader:
    """Multithreading: threads load + transform the samples of each batch in parallel."""

    def __init__(self, dataset, batch_size, num_threads, shuffle=True):
        self.dataset, self.batch_size = dataset, batch_size
        self.num_threads, self.shuffle = num_threads, shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            for i in range(0, n, self.batch_size):
                items = list(ex.map(self.dataset.__getitem__, order[i:i + self.batch_size]))
                xs = torch.stack([x for x, _ in items])
                ys = torch.tensor([y for _, y in items])
                yield xs, ys


def make_loader(dataset, method, workers):
    if method == "Sequential":
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    if method == "Multiprocessing":
        # persistent_workers=True: worker processes are started once per run and reused across
        # epochs (on Windows, starting workers is slow, so restarting them every epoch would
        # dominate the timings).
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=workers, pin_memory=(DEVICE == "cuda"),
                          persistent_workers=True)
    if method == "Multithreading":
        return ThreadedLoader(dataset, BATCH_SIZE, workers)
    raise ValueError(method)


# ---------------------------------------------------------- resource monitor
class ResourceMonitor:
    """Samples CPU %, RAM (this process + workers) and GPU % in a background thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.cpu, self.ram, self.gpu = [], [], []
        self._stop = threading.Event()

    def _rss_mb(self):
        proc = psutil.Process()
        total = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:
                pass
        return total / 1e6

    def _loop(self):
        psutil.cpu_percent(None)
        while not self._stop.is_set():
            time.sleep(self.interval)
            self.cpu.append(psutil.cpu_percent(None))
            self.ram.append(self._rss_mb())
            if GPU_HANDLE is not None:
                try:
                    self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu)
                except Exception:
                    pass

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._t.join()

    def summary(self):
        return {
            "CPU Usage (%)": float(np.mean(self.cpu)) if self.cpu else np.nan,
            "RAM Peak (MB)": float(np.max(self.ram)) if self.ram else np.nan,
            "GPU Usage (%)": float(np.mean(self.gpu)) if self.gpu else np.nan,
        }


# ---------------------------------------------------------------- experiment
def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def run_experiment(dataset, num_classes, method, workers, epochs):
    loader = make_loader(dataset, method, workers)
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    epoch_times, load_time = [], 0.0

    with ResourceMonitor() as mon:
        sync()
        t_start = time.perf_counter()
        for _ in range(epochs):
            sync()
            t_ep = time.perf_counter()
            it = iter(loader)
            while True:
                t0 = time.perf_counter()
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                load_time += time.perf_counter() - t0  # time spent waiting for data
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                loss_fn(model(x), y).backward()
                opt.step()
            sync()
            epoch_times.append(time.perf_counter() - t_ep)
        total = time.perf_counter() - t_start

    del loader, model, opt
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    res = {
        "Method": method,
        "Workers": workers,
        "Time/Epoch (s)": float(np.mean(epoch_times)),
        "Total Time (s)": total,
        "Data Loading Time (s)": load_time,
        "Throughput (img/s)": len(dataset) * epochs / total,
    }
    res.update(mon.summary())
    return res


# -------------------------------------------------------------------- graphs
def make_graphs(results):
    import matplotlib
    matplotlib.use("Agg")  # save to files, no window needed
    import matplotlib.pyplot as plt

    seq = results[results.Method == "Sequential"].iloc[0]
    thr = results[results.Method == "Multithreading"]
    prc = results[results.Method == "Multiprocessing"]

    def plot(col, ylabel, title, fname):
        plt.figure(figsize=(7, 4.5))
        plt.axhline(seq[col], color="gray", linestyle="--", label="Sequential (0 workers)")
        plt.plot(thr["Workers"], thr[col], marker="o", label="Multithreading")
        plt.plot(prc["Workers"], prc[col], marker="s", label="Multiprocessing")
        plt.xlabel("Number of workers")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
        plt.close()

    plot("Total Time (s)", "Total training time (s)", "Execution Time vs Number of Workers", "time_vs_workers.png")
    plot("Throughput (img/s)", "Throughput (images/s)", "Throughput vs Number of Workers", "throughput_vs_workers.png")
    if results["GPU Usage (%)"].notna().any():
        plot("GPU Usage (%)", "Average GPU utilization (%)", "GPU Utilization vs Number of Workers", "gpu_vs_workers.png")


# ---------------------------------------------------------------------- main
def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    parser.add_argument("--quick", action="store_true",
                        help="tiny test run (500 samples, 1 epoch, 1 repeat, fewer worker counts)")
    parser.add_argument("--tag", default="",
                        help="name for this machine/person, e.g. 'laptop-rtx4060'. "
                             "Results are saved to results/<tag>/ so runs on different PCs don't overwrite each other")
    args = parser.parse_args()
    if args.tag:
        OUT_DIR = os.path.join(OUT_DIR, "results", args.tag)
        os.makedirs(OUT_DIR, exist_ok=True)

    max_samples, epochs, repeats = MAX_SAMPLES, EPOCHS, REPEATS
    thread_counts, process_counts = THREAD_COUNTS, PROCESS_COUNTS
    if args.quick:
        max_samples, epochs, repeats = 500, 1, 1
        thread_counts, process_counts = [2], [1, 2]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset, num_classes, classes = build_dataset(args.data_dir, max_samples)
    print("Device:", DEVICE)
    print("Classes:", classes)
    print("Dataset size used:", len(dataset))

    # Hardware specs (required in the report)
    specs = get_specs(len(dataset), BATCH_SIZE, epochs)
    lines = [f"{k}: {v}" for k, v in specs.items()]
    print("\n".join(lines))
    with open(os.path.join(OUT_DIR, "hardware_specs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Warm-up (initialises CUDA / cuDNN so the first run isn't unfairly slow)
    print("\nWarm-up run...")
    run_experiment(dataset, num_classes, "Sequential", 0, 1)

    configs = [("Sequential", 0)]
    configs += [("Multithreading", n) for n in thread_counts]
    configs += [("Multiprocessing", n) for n in process_counts]

    runs = []
    for method, workers in configs:
        for r in range(repeats):
            print(f"Running {method} | workers={workers} | run {r + 1}/{repeats}")
            runs.append(run_experiment(dataset, num_classes, method, workers, epochs))

    raw = pd.DataFrame(runs)
    results = raw.groupby(["Method", "Workers"], sort=False).mean(numeric_only=True).reset_index()
    base = results.loc[results.Method == "Sequential", "Total Time (s)"].iloc[0]
    results["Speedup"] = base / results["Total Time (s)"]
    results = results.round(2)

    raw.to_csv(os.path.join(OUT_DIR, "raw_runs.csv"), index=False)
    results.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    make_graphs(results)

    print("\n=== RESULTS ===")
    print(results.to_string(index=False))
    print(f"\nSaved results.csv, raw_runs.csv, hardware_specs.txt and graphs in: {OUT_DIR}")


# This guard is REQUIRED on Windows/Mac, otherwise multiprocessing workers crash.
if __name__ == "__main__":
    main()
