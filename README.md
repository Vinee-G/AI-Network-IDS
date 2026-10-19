# AI-based Network Intrusion Detection System (NIDS)

Trains and compares **Random Forest, XGBoost and a small neural network** that classify network connections as *Normal* or as an attack family (DoS, Probe, R2L, U2R for NSL-KDD; DoS, DDoS, PortScan, BruteForce, WebAttack, Bot for CICIDS2017), then shows everything in a **web dashboard**.

**Cost: ₹0.** Everything runs on a normal laptop CPU with free open-source tools. No GPU, no paid service.

| | Practice data (built in) | NSL-KDD | CICIDS2017 |
|---|---|---|---|
| Download | none (`make_synthetic.py`) | small (a few MB) | large (about 1 GB or more) |
| Records | 38,000 | about 125k train / 22k test | about 2.8 million |
| Features | 41 | 41 | about 78 flow features |
| Laptop needed | any | any (4 GB RAM) | 8 GB RAM recommended, use `--sample` |
| Realism | low (for learning the pipeline) | old, but the standard classroom benchmark | newer, closer to modern traffic |

**Suggested path:** practice data first (Steps 1-6), then real NSL-KDD (Step 7), then CICIDS2017 as the upgrade (Step 8).

## How it works
```
 NSL-KDD files / CICIDS2017 CSVs / practice data
        |  nidslib/data.py      loads + cleans (inf, NaN, duplicates, id columns), maps attack names to families
        v
 oversample rare classes (TRAINING data only)
        |  nidslib/models.py    log-scale heavy-tailed columns, scale numbers, one-hot categories
        v
 Random Forest | XGBoost | Neural network (MLP 64-32)
        |  nidslib/evaluate.py  macro-F1, per-class recall, detection rate, false-alarm rate, unseen attacks
        v
 results/results.json + models/*.joblib  ->  dashboard.py (web page)  and  predict.py (command line)
```

## Folder structure
```
make_synthetic.py   practice data in the real NSL-KDD file format
train.py            trains and compares the 3 models, saves results
dashboard.py        web dashboard (Flask)            templates/dashboard.html
predict.py          classify any CSV with the best model
nidslib/            schema.py, data.py, models.py, evaluate.py
tests/test_sanity.py
```

---

## Step-by-step guide

### Step 1 - Install (10 min)
Install Python 3.10+ and VS Code. In this folder:
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / Mac
pip install -r requirements.txt
```
If `pip install xgboost` fails, the project still runs: it uses scikit-learn's HistGradientBoosting and labels it clearly. Install xgboost to get the real XGBoost.

### Step 2 - Create practice data (1 min)
```bash
python make_synthetic.py
```
Creates `data/synthetic_nsl/KDDTrain+.txt` and `KDDTest+.txt` in the exact NSL-KDD format. The test file contains attacks that never appear in training (apache2, mscan, snmpguess, sqlattack), like the real KDDTest+.

### Step 3 - Train and compare the models (1 to 3 min)
```bash
python train.py
```
You get a table like this (your numbers may differ slightly):

| Column | Meaning |
|---|---|
| macroF1 | F1 averaged over classes, so rare attacks count as much as common ones |
| acc | Overall accuracy (misleading when one class dominates) |
| DR | Detection rate: attacks caught as any attack |
| FAR | False-alarm rate: normal traffic wrongly flagged |
| train s / ms/1k | Training time, and prediction time per 1,000 records |

Take a screenshot for your report.

### Step 4 - Open the dashboard
```bash
python dashboard.py
```
Open http://127.0.0.1:5000. You will see the model comparison, a confusion matrix, per-class recall, top features, seen vs unseen attacks, and a live replay that classifies random test records with all models side by side.

### Step 5 - Classify your own file
```bash
python predict.py --csv data/synthetic_nsl/KDDTest+.txt
python predict.py --csv my_flows.csv --model "Random Forest"
```
Writes `results/predictions.csv`. CSVs need a header row with the same feature names (NSL-KDD files without a header also work).

### Step 6 - Run the tests
```bash
python tests/test_sanity.py
```

### Step 7 - Train on REAL NSL-KDD (do this before quoting any number)
1. Download NSL-KDD from the University of New Brunswick (CIC) dataset page, or a mirror on GitHub or Kaggle. Check the current download page; links and mirrors change.
2. Create `data/nsl_kdd/` and put `KDDTrain+.txt` and `KDDTest+.txt` inside.
3. Run:
```bash
python train.py --dataset nsl --path data/nsl_kdd
```
Then refresh the dashboard. The yellow "synthetic data" banner disappears.

Expect lower scores than on practice data, mainly on R2L and U2R, and on attacks that appear only in the test file. That is normal and is good material for your report.

Tips: `--max-train 40000` for a faster run; `--binary` for Normal vs Attack only. The harder `KDDTest-21` file (if you download it) is a nice extra experiment: name it `KDDTest+.txt` in another folder and compare.

### Step 8 - Upgrade to CICIDS2017
1. Download the "MachineLearningCSV" files from the UNB CIC CICIDS2017 page. They are several CSV files (Monday to Friday). Check the page for the current download method.
2. Put all CSVs in `data/cicids2017/`.
3. Run:
```bash
python train.py --dataset cicids --path data/cicids2017 --sample 150000
```
What the loader does automatically: trims column names, drops identifier columns (Flow ID, IPs, Timestamp, Source Port), removes `inf`/NaN/duplicate rows, groups the 15 labels into 7 families, and takes a **class-capped sample** to keep it cheap. Rare classes dropped if under 10 rows.

Important caveats (also good interview points):
- The split is random, and flows from the same attack session end up on both sides, so scores can be unrealistically high (often above 99%).
- Class-capped sampling changes the class mix, so the false-alarm rate is not what a real network would see.
- Researchers have reported labeling and feature-extraction problems in the original CICIDS2017 files; mention it in your report.
- For rare classes, the default oversampling target (3,000 rows) can duplicate a tiny class many times. Try `--balance-target 1000` or `--no-balance` and compare.

### Step 9 - Experiments that make the project stand out
Pick two or three and put the results in your report:
1. **Unseen attacks:** Which model handles new attacks best? (Dashboard section "Seen vs unseen".)
2. **With vs without oversampling:** `--no-balance`. What happens to R2L and U2R recall?
3. **Detection vs false alarms:** Change the decision rule, for example only flag an attack when its probability is above 0.7, and plot DR against FAR.
4. **Feature selection:** Train with only the top 10 features from the Random Forest. How much accuracy do you lose, and how much faster is it?
5. **Binary vs multi-class:** `--binary`. Is "Normal or Attack" easier?
6. **Time-based or attack-held-out split** on CICIDS2017 to remove the session leakage.
7. **Tune** the models with `RandomizedSearchCV` (free, a few minutes).

### Step 10 - Write it up
Include: problem statement, dataset description, preprocessing, model comparison table, confusion matrices, unseen-attack results, limitations, and future work. Put the code on GitHub with dashboard screenshots.

**Portfolio idea:** feed this NIDS output into your SOC alert-triage project. NIDS detects, the triage tool ranks. That tells one clear story.

---

## Reading your results honestly
- **Accuracy lies when classes are imbalanced.** A model that says "Normal" for everything can score above 50% on NSL-KDD. Use macro-F1 and per-class recall.
- **False alarms matter in practice.** A 1% false-alarm rate sounds small, but on 1 million normal connections per day that is 10,000 false alerts. Always discuss detection rate and false-alarm rate together.
- **Practice data is not real.** The dashboard banner says so. Do not quote its scores as real-world results.
- **NSL-KDD is old** (derived from 1999 traffic) and does not look like modern traffic. It is a benchmark for learning, not proof of production readiness.
- **A model is a detector, not a defence.** In real use it sits behind a flow extractor (for example CICFlowMeter) and feeds a SOC.

## Resume bullets (replace the numbers with your real-data results)
- Built an ML-based network intrusion detection system on NSL-KDD and CICIDS2017, comparing Random Forest, XGBoost and a neural network (MLP) with macro-F1, per-class recall, detection rate and false-alarm rate.
- Handled class imbalance (oversampling of rare attacks on training data only) and evaluated generalisation to attack types unseen during training.
- Developed a Flask dashboard showing confusion matrices, feature importance and a live side-by-side replay of model predictions.
- Engineered a reproducible pipeline (data cleaning for inf/NaN/duplicates, scaling, one-hot encoding, saved models) with automated tests.

## Interview questions to prepare
1. **Why macro-F1 and not accuracy?** Class imbalance: rare attacks would be ignored by accuracy.
2. **Why oversample only the training set?** Oversampling the test set (or before splitting) leaks duplicated rows into evaluation and inflates scores.
3. **Why do models do worse on unseen attacks?** They learn the signatures of known attacks; anomaly-based or semi-supervised methods can help.
4. **What is wrong with a random split on CICIDS2017?** Near-duplicate flows from one attack session appear in both train and test.
5. **Trees or neural network for tabular data?** Trees often match or beat small neural networks on tabular data, are faster to train, and give feature importance.
6. **How would you deploy it?** Capture traffic, extract flow features, score in batches or streaming, send alerts to a SIEM, monitor drift and retrain.
7. **How do you control alert fatigue?** Threshold tuning, grouping alerts, and prioritisation (your triage tool).

## Troubleshooting
| Problem | Fix |
|---|---|
| `FileNotFoundError ... KDDTrain+.txt` | Run `python make_synthetic.py`, or check the folder you pass with `--path` |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |
| Dashboard says "No results found" | Run `python train.py` first |
| CICIDS2017 runs out of memory | Lower `--sample` (for example 80000) |
| XGBoost not installed | `pip install xgboost`, then rerun `train.py` |
| Training is slow | Use `--max-train` (NSL-KDD) or a smaller `--sample` (CICIDS2017) |

## Legal and ethical note
Use only public research datasets and traffic from your own lab. Do not capture or analyse other people's network traffic without permission.
