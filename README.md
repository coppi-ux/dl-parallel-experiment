# Multithreading vs Multiprocessing in a Deep Learning Pipeline

Group assignment: measure how sequential, multithreaded and multiprocess data loading affect
training time, throughput and CPU / RAM / GPU usage.

- **Task:** image classification (6 classes) on the Intel Image Classification dataset
- **Model:** ResNet18 (trained from scratch, accuracy is not the goal)
- **Scenarios:** Sequential (0 workers), Multithreading (2/4/8 threads), Multiprocessing (1/2/4/8 workers)
- **Measured:** total time, time per epoch, data loading time, throughput, CPU %, peak RAM, GPU %, speedup

Everything is in one script: `dl_parallel_experiment.py`.

## Reference results (from the team lead's laptop)

Setup: 10,000 images, batch size 64, 3 epochs, each configuration run 2 times and averaged.
Hardware: Intel CPU (10 cores / 16 threads), 34 GB RAM, NVIDIA RTX 4060 Laptop GPU, Windows 11,
Python 3.13.5, PyTorch 2.11.0+cu128.

| Method | Workers | Total Time (s) | Data Loading Time (s) | Throughput (img/s) | GPU Usage (%) | Speedup |
|---|---|---|---|---|---|---|
| Sequential | 0 | 200.34 | 187.89 | 150.23 | 18.21 | 1.00 |
| Multithreading | 2 | 213.76 | 191.71 | 141.04 | 20.75 | 0.94 |
| Multithreading | 4 | 206.51 | 183.45 | 145.51 | 18.97 | 0.97 |
| Multithreading | 8 | 199.40 | 176.51 | 150.46 | 21.58 | 1.00 |
| Multiprocessing | 1 | 186.63 | 167.23 | 160.75 | 20.95 | 1.07 |
| Multiprocessing | 2 | 105.38 | 78.11 | 284.71 | 20.06 | 1.90 |
| Multiprocessing | 4 | 81.37 | 35.97 | 368.69 | 22.13 | **2.46** |
| Multiprocessing | 8 | 95.87 | 13.38 | 312.95 | 18.54 | 2.09 |

Full data for this machine is in `results/` (results.csv, raw_runs.csv, hardware_specs.txt, graphs).

## Run it on your own PC

Speedup is only comparable **within one machine**, so always compare your own sequential
result with your own multiprocessing result. Don't mix numbers from different PCs in one table.

### 1. Get the code
```
git clone <REPO-URL>
cd <REPO-FOLDER>
```

### 2. Create a virtual environment
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\activate
```
Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```
The prompt should start with `(.venv)`.

### 3. Install PyTorch and the other packages
- **NVIDIA GPU:** get the command for your CUDA version from https://pytorch.org/get-started/locally/
  (for example `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`).
- **No NVIDIA GPU:** `pip install torch torchvision` (training will run on the CPU, which is much slower).

Then:
```
pip install pandas matplotlib psutil pynvml pillow
```
Check the GPU: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### 4. Download the dataset
Download **Intel Image Classification** from
https://www.kaggle.com/datasets/puneet6060/intel-image-classification (free Kaggle account needed)
and unzip it into a folder called `data/` inside the project.
The class folders should be at `data/seg_train/seg_train` (buildings, forest, glacier, mountain, sea, street).
Check with `dir data\seg_train\seg_train` (Windows) or `ls data/seg_train/seg_train` (Mac/Linux).

### 5. Quick test first (about 1-3 minutes)
```
python dl_parallel_experiment.py --data-dir data/seg_train/seg_train --quick --tag my-name
```

### 6. Full run
```
python dl_parallel_experiment.py --data-dir data/seg_train/seg_train --tag my-name
```
- Plug in the charger and close heavy programs (browser, games, Zoom). Don't use the PC while it runs.
- **No NVIDIA GPU?** The full run can take hours. Lower `MAX_SAMPLES` (for example 2000) and
  `EPOCHS` (for example 2) in the CONFIG section at the top of the script.
- Set `PROCESS_COUNTS` / `THREAD_COUNTS` at the top of the script to match your CPU if needed.

Your results are saved in `results/my-name/`: `hardware_specs.txt`, `results.csv`, `raw_runs.csv`
and the graphs (`time_vs_workers.png`, `throughput_vs_workers.png`, `gpu_vs_workers.png`).

### 7. Share your results
```
git checkout -b my-name
git add results/my-name
git commit -m "Add results from my PC"
git push origin my-name
```
Or just send the `results/my-name` folder to the group.

## Troubleshooting
- `No module named 'numpy'` (or another package): the wrong environment is active. Run `deactivate`, then
  activate `.venv` again (Windows: `.venv\Scripts\activate`).
- PowerShell says scripts are disabled: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
- The `FutureWarning` about `pynvml` is harmless.
- `Device: cpu` printed but you have an NVIDIA GPU: you installed the CPU-only PyTorch. Uninstall it
  (`pip uninstall -y torch torchvision`) and install the CUDA version from step 3.
