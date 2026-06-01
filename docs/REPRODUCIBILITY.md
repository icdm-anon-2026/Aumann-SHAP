\## Execution order (scripts depend on files created earlier)



\### German Credit

All German scripts use `CACHE\_DIR = "./cache"` (relative to `experiments/german\_credit/`).



\*\*Required pretrained models (no training here):\*\*

\- `experiments/german\_credit/cache/models\_split\_rs1.joblib`  (required by `local`, `within\_pot`, `global`, `msweep`)



\*\*Order:\*\*

1\) `python experiments/run\_german\_credit.py --task local`

&nbsp;  - creates (if missing): `cache/cf\_t080\_bad30\_idx\_242.json`

&nbsp;  - creates: `cache/artifact\_rs1\_thr30\_t080.json` (master artifact)

&nbsp;  - also writes: `cache/totals\_\*\_rs1\_thr30\_t080.csv`, `cache/OUTPUT\_\*.txt`, `cache/artifact\_\*.json`



2\) `python experiments/run\_german\_credit.py --task within\_pot`

&nbsp;  - reads: `cache/models\_split\_rs1.joblib` + `cache/cf\_t080\_bad30\_idx\_242.json`

&nbsp;  - creates: `cache/within\_pot\_summary\_\*.csv`, `cache/within\_pot\_long\_\*.csv`



3\) `python experiments/run\_german\_credit.py --task global`

&nbsp;  - reads: `cache/models\_split\_rs1.joblib`

&nbsp;  - creates: `cache/global\_meta\_\*.csv`, `cache/global\_long\_\*.csv`, `cache/global\_avg\_\*.csv`,

&nbsp;            `cache/global\_cf\_endpoints\_\*.jsonl`, `cache/global\_artifact\_\*.json`, `cache/OUTPUT\_\*.txt`



4\) `python experiments/run\_german\_credit.py --task msweep`

&nbsp;  - reads: `cache/artifact\_rs1\_thr30\_t080.json` + `cache/models\_split\_rs1.joblib`

&nbsp;  - creates: `cache/msweep\_long\_idx\*\_rs1\_thr30\_t080.csv`, `cache/msweep\_final\_idx\*\_rs1\_thr30\_t080.csv`



5\) `python experiments/run\_german\_credit.py --task convergence`

&nbsp;  - reads: a CSV in `cache/` (see `CSV\_NAME` inside `convergence.py`, typically the `msweep\_final\_\*.csv`)

&nbsp;  - creates: convergence plot(s) (see `convergence.py`)



> Note: `within\_pot.py` is hard-coded to `cf\_t080\_bad30\_idx\_242.json`. If you change the idx/target/thr in `local\_analysis.py`,

> update `CF\_CACHE\_FILE` in `within\_pot.py` accordingly.



---



\### MNIST

All MNIST scripts run inside `experiments/mnist/`.



\*\*Heavy step (only if missing):\*\*

\- `train` creates the checkpoint: `resnet18\_mnist\_1vs7.pt`



\*\*Order (because of caches):\*\*

1\) `python experiments/run\_mnist.py --task train`

&nbsp;  - creates: `resnet18\_mnist\_1vs7.pt`



2\) `python experiments/run\_mnist.py --task equal\_split`

&nbsp;  - reads: `resnet18\_mnist\_1vs7.pt`

&nbsp;  - creates: `cache\_attribs/eqsplit\_idx2\_to\_1809\_eps0.05\_perms400.pt` (+ `\*\_heat.npy`)



3\) `python experiments/run\_mnist.py --task micro\_game`

&nbsp;  - reads: `resnet18\_mnist\_1vs7.pt`

&nbsp;  - creates: `cache\_attribs/microgame\_idx2\_to\_1809\_eps0.05\_m10\_perms200.pt` (+ `\*\_heat.npy`)



4\) `python experiments/run\_mnist.py --task heatmaps`

&nbsp;  - reads: the two `.pt` files above (hard-coded in `heatmaps.py`)

&nbsp;  - creates: `figures/mnist\_1to7\_all\_maps\_4x4.png`



5\) `python experiments/run\_mnist.py --task patchtest`

&nbsp;  - reads: latest matching eqsplit/microgame caches in `cache\_attribs/` (and also loads `resnet18\_mnist\_1vs7.pt`)

&nbsp;  - creates: patchtest figure(s) in `figures/`



6\) `python experiments/run\_mnist.py --task global`

&nbsp;  - reads: `resnet18\_mnist\_1vs7.pt`

&nbsp;  - creates: `cache\_global/\*.pt` + `\*\_mean\_abs\_\*.npy` and figures in `figures\_global/`



7\) `python experiments/run\_mnist.py --task globalheat`

&nbsp;  - reads: latest `cache\_global/\*.pt`

&nbsp;  - creates: global heat figures (see prints in `globalheat.py`)

