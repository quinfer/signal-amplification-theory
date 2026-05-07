# Source-data metadata probe

Generated at 2026-05-02T18:04:59

This file is produced by `qf/probe_metadata.py`.  It introspects the candidate source-data files for any timestamp or stock-identifier metadata that the chronological-split harness can use.  It does **not** modify the source data.

## Candidate files

### `x_data.pkl`
- path: `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection/data/x_data.pkl`
- size: 462.29 MB
- top-level type: `list`
- `x_data.pkl`: list, length=43050
  - element[0] is ndarray, shape=(51, 26), dtype=float64

#### Episode-window probe (n_episodes=43050, shape[0]=(51, 26))
- window 0: shape=(51, 26), dtype=float64
- window 1: shape=(51, 26), dtype=float64
- window 2: shape=(52, 26), dtype=float64

### `y_dex.pkl`
- path: `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection/data/y_dex.pkl`
- size: 0.04 MB
- top-level type: `ndarray`
- `y_dex.pkl`: ndarray, shape=(43050,), dtype=bool
  - first 3: False, False, False

### `data_1115.pkl`
- path: `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection/Documents and Data/data_1115.pkl`
- size: 9.14 MB
- top-level type: `DataFrame`
- `data_1115.pkl`: DataFrame, shape=(142769, 8)
  - columns: ['BS_deal', 'Time_start', 'Time_end', 'Price_1', 'Price_2', 'Volume_1', 'Volume_2', 'Volume']
  - dtypes: {'BS_deal': dtype('int64'), 'Time_start': dtype('int64'), 'Time_end': dtype('int64'), 'Price_1': dtype('int64'), 'Price_2': dtype('int64'), 'Volume_1': dtype('int64'), 'Volume_2': dtype('int64'), 'Volume': dtype('int64')}

### `000001.csv`
- path: `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/chinese market manipulation cases (sample)/000001.csv`
- size: 6.41 MB
- columns: ['TranID', 'Time', 'Price', 'Volume', 'SaleOrderVolume', 'BuyOrderVolume', 'Type', 'SaleOrderID', 'SaleOrderPrice', 'BuyOrderID', 'BuyOrderPrice']
- dtypes: {'TranID': dtype('int64'), 'Time': dtype('O'), 'Price': dtype('float64'), 'Volume': dtype('int64'), 'SaleOrderVolume': dtype('int64'), 'BuyOrderVolume': dtype('int64'), 'Type': dtype('O'), 'SaleOrderID': dtype('int64'), 'SaleOrderPrice': dtype('float64'), 'BuyOrderID': dtype('int64'), 'BuyOrderPrice': dtype('float64')}
- first 5 rows:

```
 TranID     Time  Price  Volume  SaleOrderVolume  BuyOrderVolume Type  SaleOrderID  SaleOrderPrice  BuyOrderID  BuyOrderPrice
      1 09:25:00  22.31     100              400             100    B            1           20.13           1          24.61
      2 09:25:00  22.31     300              400            1200    B            1           20.13           2          24.61
      3 09:25:00  22.31     400              400            1200    B            3           20.13           2          24.61
      4 09:25:00  22.31     100              100            1200    B            4           20.13           2          24.61
      5 09:25:00  22.31     400            13600            1200    B            5           20.13           2          24.61
```
- Time range in first 200,000 rows: 2026-05-02 09:25:00 to 2026-05-02 15:00:00

## What to look for

The chronological-split harness needs a per-episode date that is comparable across episodes.  Likely sources, in order of preference:

1. A datetime column inside each LOB-window ndarray inside `x_data.pkl`. If present, it allows the *exact* episode date (and time-of-day) to be recovered without reaching outside the existing pipeline.
2. A separate metadata structure inside `data_1115.pkl` aligned to the same episode index as `y_dex.pkl`.  If `data_1115.pkl` is a `dict` with keys like `episode_id`, `date`, or `stock_id`, this is the cleanest path.
3. The CSRC PDF folder.  Each penalty decision typically lists the manipulation date(s); these can be parsed into a per-case date table that is then matched to the labelled episodes via stock id and a date proximity rule.

Once the date source is identified, the next step is to fill in `get_episode_dates()` in `qf/jfqa_chrono.py` and run the chronological harness.
---

# Deeper metadata probe (appended)

Generated at 2026-05-02T18:10:56

### Deeper LOB-window column probe

Pooled column statistics across the first 200 episodes of ``x_data.pkl`` (53 row-snapshots each, on average).

- per-window shape: (51, 26); n_columns = 26

```
column     n  min       max      mean       std  n_unique monotone_inside_episode
 col_0 10206    0      5.26     4.145    0.8592       171                      no
 col_1 10206    0      5.21     4.121    0.8787       167                      no
 col_2 10206    0      5.23     4.128    0.8803       169                      no
 col_3 10206    0 4.329e+06 6.638e+04 1.158e+05      5071                      no
 col_4 10206    0 2.055e+07  2.96e+05  5.46e+05      9579                      no
 col_5 10206    0      5.23     4.138    0.8575       169                      no
 col_6 10206    0      5.22     4.276    0.3839       167                      no
 col_7 10206    0      5.21     4.134    0.8262       167                      no
 col_8 10206    0       5.2     4.124    0.8246       167                      no
 col_9 10206    0      5.19     4.115     0.823       167                      no
col_10 10206    0      5.18     4.105    0.8214       167                      no
col_11 10206    0 1.493e+06 4.391e+04 6.679e+04      5244                      no
col_12 10206    0 1.803e+06 7.398e+04 9.543e+04      4553                      no
col_13 10206    0   1.8e+06 7.644e+04 8.926e+04      3376                      no
col_14 10206    0   1.8e+06 7.463e+04 9.715e+04      2674                      no
col_15 10206    0 1.808e+06 7.006e+04 1.084e+05      2275                      no
col_16 10206    0      5.23     4.287    0.3837       169                      no
col_17 10206    0      5.24     4.165    0.8311       168                      no
col_18 10206    0      5.25     4.175    0.8327       168                      no
col_19 10206    0      5.26     4.184    0.8343       167                      no
col_20 10206    0      5.27     4.194    0.8359       167                      no
col_21 10206    0 9.693e+05 4.538e+04 5.601e+04      5509                      no
col_22 10206    0 1.255e+06 7.386e+04 7.733e+04      5134                      no
col_23 10206    0 1.257e+06 7.774e+04 7.525e+04      4246                      no
col_24 10206    0 1.255e+06 7.612e+04 7.092e+04      3697                      no
col_25 10206    0 1.255e+06 7.282e+04 7.098e+04      3423                      no
```

#### Per-column heuristic interpretation

- `col_0` [0, 5.26], n_unique=171, monotone=no: no obvious interpretation
- `col_1` [0, 5.21], n_unique=167, monotone=no: no obvious interpretation
- `col_2` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_3` [0, 4.33e+06], n_unique=5071, monotone=no: range fits trade volume
- `col_4` [0, 2.05e+07], n_unique=9579, monotone=no: range fits trade volume
- `col_5` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_6` [0, 5.22], n_unique=167, monotone=no: no obvious interpretation
- `col_7` [0, 5.21], n_unique=167, monotone=no: no obvious interpretation
- `col_8` [0, 5.2], n_unique=167, monotone=no: no obvious interpretation
- `col_9` [0, 5.19], n_unique=167, monotone=no: no obvious interpretation
- `col_10` [0, 5.18], n_unique=167, monotone=no: no obvious interpretation
- `col_11` [0, 1.49e+06], n_unique=5244, monotone=no: range fits trade volume
- `col_12` [0, 1.8e+06], n_unique=4553, monotone=no: range fits trade volume
- `col_13` [0, 1.8e+06], n_unique=3376, monotone=no: range fits trade volume
- `col_14` [0, 1.8e+06], n_unique=2674, monotone=no: range fits trade volume
- `col_15` [0, 1.81e+06], n_unique=2275, monotone=no: range fits trade volume
- `col_16` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_17` [0, 5.24], n_unique=168, monotone=no: no obvious interpretation
- `col_18` [0, 5.25], n_unique=168, monotone=no: no obvious interpretation
- `col_19` [0, 5.26], n_unique=167, monotone=no: no obvious interpretation
- `col_20` [0, 5.27], n_unique=167, monotone=no: no obvious interpretation
- `col_21` [0, 9.69e+05], n_unique=5509, monotone=no: range fits trade volume
- `col_22` [0, 1.25e+06], n_unique=5134, monotone=no: range fits trade volume
- `col_23` [0, 1.26e+06], n_unique=4246, monotone=no: range fits trade volume
- `col_24` [0, 1.25e+06], n_unique=3697, monotone=no: range fits trade volume
- `col_25` [0, 1.25e+06], n_unique=3423, monotone=no: range fits trade volume

#### First three rows of two windows (for eyeballing)

window 0, shape (51, 26):
```
[[     0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.
       0.        0.        0.        0.        0.        0.        0.        0.        0.  ]
 [     0.        0.        0.        0.        0.        0.        3.9       0.        0.        0.        0.    26500.    29800.        0.        0.        0.        3.9
       0.        0.        0.        0.    26500.        0.        0.        0.        0.  ]
 [     3.91      0.        0.    26500.   103615.        3.91      3.91      3.9       3.89      3.88      3.87   6100.    32300.     3500.     6500.     3800.        3.92
       3.93      3.94      3.95      3.96   8500.     1800.    33400.    11800.   110900.  ]]
```

window 1, shape (51, 26):
```
[[     0.        0.        0.        0.        0.        0.        3.87      0.        0.        0.        0.     1200.     7000.        0.        0.        0.        3.87
       0.        0.        0.        0.     1200.        0.        0.        0.        0.  ]
 [     0.        0.        0.        0.        0.        0.        3.87      0.        0.        0.        0.     2900.    60900.        0.        0.        0.        3.87
       0.        0.        0.        0.     2900.        0.        0.        0.        0.  ]
 [     3.87      3.87      3.87   2900.    11223.        3.87      3.87      3.86      3.85      3.84      3.83  60900.   108600.   221800.    14600.     4800.        3.88
       3.89      3.9       3.91      3.92    300.     8800.    20200.    35700.    35000.  ]]
```

#### Cross-episode constancy check

For each column we report the standard deviation of the per-episode mean.  A column whose per-episode mean varies a lot is more likely to be a feature; a column whose per-episode mean is concentrated on a small set of values may be a date/stock identifier.

```
column  between_episode_std  between_episode_n_unique  between_episode_min  between_episode_max
 col_0               0.3397                        92                3.482                5.113
 col_1               0.3398                       100                  3.4                5.086
 col_2               0.3406                        97                3.404                5.097
 col_3            5.791e+04                       200            1.184e+04            5.277e+05
 col_4            2.812e+05                       200            4.866e+04            2.438e+06
 col_5               0.3386                        98                3.475                  5.1
 col_6               0.3539                        98                 3.57                5.156
 col_7               0.3356                        95                3.463                5.083
 col_8               0.3356                        96                3.453                5.073
 col_9               0.3356                        96                3.444                5.063
col_10               0.3356                        97                3.434                5.053
col_11            2.791e+04                       200            1.022e+04            2.846e+05
col_12            4.908e+04                       200            1.676e+04             4.39e+05
col_13            4.834e+04                       200            1.361e+04            3.988e+05
col_14            5.469e+04                       200            1.549e+04            4.298e+05
col_15            6.274e+04                       200                 8432            4.481e+05
col_16               0.3535                        99                3.582                5.167
col_17               0.3352                        93                3.493                5.116
col_18               0.3352                        93                3.503                5.126
col_19               0.3351                        92                3.512                5.136
col_20               0.3352                        93                3.522                5.146
col_21            2.479e+04                       200            1.046e+04             1.66e+05
col_22            4.622e+04                       200            1.698e+04            2.479e+05
col_23            4.658e+04                       200            1.282e+04            2.344e+05
col_24            4.429e+04                       200            1.057e+04            2.593e+05
col_25             4.42e+04                       200            1.295e+04            2.877e+05
```

### Deeper inspection of `Time_start` / `Time_end` in `data_1115.pkl`

#### `Time_start`

- min = 92959000, max = 145958030
- mean = 112581136, n_unique = 65413
- first 10 raw: [92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000]
- last 10 raw : [145642010, 145652710, 145654290, 145654850, 145656340, 145657030, 145657140, 145657490, 145701500, 145722590]
- **no standard time encoding fits this range**

#### `Time_end`

- min = 93000000, max = 150000000
- mean = 114189491, n_unique = 68223
- first 10 raw: [93000000, 93000000, 93000010, 93000010, 93000010, 93000010, 93000010, 93000010, 93000020, 93000020]
- last 10 raw : [150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000]
- **no standard time encoding fits this range**


### Sibling files in `market_manipulation_detection/`

(searching for build scripts and metadata sidecars)

```
       9138237 bytes  Documents and Data/data_1115.pkl
         85500 bytes  FinanaceModel/MLP_CNN.py
     462285382 bytes  data/x_data.pkl
         43234 bytes  data/y_dex.pkl
          7622 bytes  sample.csv
```

---

# Deeper metadata probe (appended)

Generated at 2026-05-04T09:39:18

### Deeper LOB-window column probe

Pooled column statistics across the first 200 episodes of ``x_data.pkl`` (53 row-snapshots each, on average).

- per-window shape: (51, 26); n_columns = 26

```
column     n  min       max      mean       std  n_unique monotone_inside_episode
 col_0 10206    0      5.26     4.145    0.8592       171                      no
 col_1 10206    0      5.21     4.121    0.8787       167                      no
 col_2 10206    0      5.23     4.128    0.8803       169                      no
 col_3 10206    0 4.329e+06 6.638e+04 1.158e+05      5071                      no
 col_4 10206    0 2.055e+07  2.96e+05  5.46e+05      9579                      no
 col_5 10206    0      5.23     4.138    0.8575       169                      no
 col_6 10206    0      5.22     4.276    0.3839       167                      no
 col_7 10206    0      5.21     4.134    0.8262       167                      no
 col_8 10206    0       5.2     4.124    0.8246       167                      no
 col_9 10206    0      5.19     4.115     0.823       167                      no
col_10 10206    0      5.18     4.105    0.8214       167                      no
col_11 10206    0 1.493e+06 4.391e+04 6.679e+04      5244                      no
col_12 10206    0 1.803e+06 7.398e+04 9.543e+04      4553                      no
col_13 10206    0   1.8e+06 7.644e+04 8.926e+04      3376                      no
col_14 10206    0   1.8e+06 7.463e+04 9.715e+04      2674                      no
col_15 10206    0 1.808e+06 7.006e+04 1.084e+05      2275                      no
col_16 10206    0      5.23     4.287    0.3837       169                      no
col_17 10206    0      5.24     4.165    0.8311       168                      no
col_18 10206    0      5.25     4.175    0.8327       168                      no
col_19 10206    0      5.26     4.184    0.8343       167                      no
col_20 10206    0      5.27     4.194    0.8359       167                      no
col_21 10206    0 9.693e+05 4.538e+04 5.601e+04      5509                      no
col_22 10206    0 1.255e+06 7.386e+04 7.733e+04      5134                      no
col_23 10206    0 1.257e+06 7.774e+04 7.525e+04      4246                      no
col_24 10206    0 1.255e+06 7.612e+04 7.092e+04      3697                      no
col_25 10206    0 1.255e+06 7.282e+04 7.098e+04      3423                      no
```

#### Per-column heuristic interpretation

- `col_0` [0, 5.26], n_unique=171, monotone=no: no obvious interpretation
- `col_1` [0, 5.21], n_unique=167, monotone=no: no obvious interpretation
- `col_2` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_3` [0, 4.33e+06], n_unique=5071, monotone=no: range fits trade volume
- `col_4` [0, 2.05e+07], n_unique=9579, monotone=no: range fits trade volume
- `col_5` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_6` [0, 5.22], n_unique=167, monotone=no: no obvious interpretation
- `col_7` [0, 5.21], n_unique=167, monotone=no: no obvious interpretation
- `col_8` [0, 5.2], n_unique=167, monotone=no: no obvious interpretation
- `col_9` [0, 5.19], n_unique=167, monotone=no: no obvious interpretation
- `col_10` [0, 5.18], n_unique=167, monotone=no: no obvious interpretation
- `col_11` [0, 1.49e+06], n_unique=5244, monotone=no: range fits trade volume
- `col_12` [0, 1.8e+06], n_unique=4553, monotone=no: range fits trade volume
- `col_13` [0, 1.8e+06], n_unique=3376, monotone=no: range fits trade volume
- `col_14` [0, 1.8e+06], n_unique=2674, monotone=no: range fits trade volume
- `col_15` [0, 1.81e+06], n_unique=2275, monotone=no: range fits trade volume
- `col_16` [0, 5.23], n_unique=169, monotone=no: no obvious interpretation
- `col_17` [0, 5.24], n_unique=168, monotone=no: no obvious interpretation
- `col_18` [0, 5.25], n_unique=168, monotone=no: no obvious interpretation
- `col_19` [0, 5.26], n_unique=167, monotone=no: no obvious interpretation
- `col_20` [0, 5.27], n_unique=167, monotone=no: no obvious interpretation
- `col_21` [0, 9.69e+05], n_unique=5509, monotone=no: range fits trade volume
- `col_22` [0, 1.25e+06], n_unique=5134, monotone=no: range fits trade volume
- `col_23` [0, 1.26e+06], n_unique=4246, monotone=no: range fits trade volume
- `col_24` [0, 1.25e+06], n_unique=3697, monotone=no: range fits trade volume
- `col_25` [0, 1.25e+06], n_unique=3423, monotone=no: range fits trade volume

#### First three rows of two windows (for eyeballing)

window 0, shape (51, 26):
```
[[     0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.
       0.        0.        0.        0.        0.        0.        0.        0.        0.  ]
 [     0.        0.        0.        0.        0.        0.        3.9       0.        0.        0.        0.    26500.    29800.        0.        0.        0.        3.9
       0.        0.        0.        0.    26500.        0.        0.        0.        0.  ]
 [     3.91      0.        0.    26500.   103615.        3.91      3.91      3.9       3.89      3.88      3.87   6100.    32300.     3500.     6500.     3800.        3.92
       3.93      3.94      3.95      3.96   8500.     1800.    33400.    11800.   110900.  ]]
```

window 1, shape (51, 26):
```
[[     0.        0.        0.        0.        0.        0.        3.87      0.        0.        0.        0.     1200.     7000.        0.        0.        0.        3.87
       0.        0.        0.        0.     1200.        0.        0.        0.        0.  ]
 [     0.        0.        0.        0.        0.        0.        3.87      0.        0.        0.        0.     2900.    60900.        0.        0.        0.        3.87
       0.        0.        0.        0.     2900.        0.        0.        0.        0.  ]
 [     3.87      3.87      3.87   2900.    11223.        3.87      3.87      3.86      3.85      3.84      3.83  60900.   108600.   221800.    14600.     4800.        3.88
       3.89      3.9       3.91      3.92    300.     8800.    20200.    35700.    35000.  ]]
```

#### Cross-episode constancy check

For each column we report the standard deviation of the per-episode mean.  A column whose per-episode mean varies a lot is more likely to be a feature; a column whose per-episode mean is concentrated on a small set of values may be a date/stock identifier.

```
column  between_episode_std  between_episode_n_unique  between_episode_min  between_episode_max
 col_0               0.3397                        92                3.482                5.113
 col_1               0.3398                       100                  3.4                5.086
 col_2               0.3406                        97                3.404                5.097
 col_3            5.791e+04                       200            1.184e+04            5.277e+05
 col_4            2.812e+05                       200            4.866e+04            2.438e+06
 col_5               0.3386                        98                3.475                  5.1
 col_6               0.3539                        98                 3.57                5.156
 col_7               0.3356                        95                3.463                5.083
 col_8               0.3356                        96                3.453                5.073
 col_9               0.3356                        96                3.444                5.063
col_10               0.3356                        97                3.434                5.053
col_11            2.791e+04                       200            1.022e+04            2.846e+05
col_12            4.908e+04                       200            1.676e+04             4.39e+05
col_13            4.834e+04                       200            1.361e+04            3.988e+05
col_14            5.469e+04                       200            1.549e+04            4.298e+05
col_15            6.274e+04                       200                 8432            4.481e+05
col_16               0.3535                        99                3.582                5.167
col_17               0.3352                        93                3.493                5.116
col_18               0.3352                        93                3.503                5.126
col_19               0.3351                        92                3.512                5.136
col_20               0.3352                        93                3.522                5.146
col_21            2.479e+04                       200            1.046e+04             1.66e+05
col_22            4.622e+04                       200            1.698e+04            2.479e+05
col_23            4.658e+04                       200            1.282e+04            2.344e+05
col_24            4.429e+04                       200            1.057e+04            2.593e+05
col_25             4.42e+04                       200            1.295e+04            2.877e+05
```

### Deeper inspection of `Time_start` / `Time_end` in `data_1115.pkl`

#### `Time_start`

- min = 92959000, max = 145958030
- mean = 112581136, n_unique = 65413
- first 10 raw: [92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000, 92959000]
- last 10 raw : [145642010, 145652710, 145654290, 145654850, 145656340, 145657030, 145657140, 145657490, 145701500, 145722590]
- **no standard time encoding fits this range**

#### `Time_end`

- min = 93000000, max = 150000000
- mean = 114189491, n_unique = 68223
- first 10 raw: [93000000, 93000000, 93000010, 93000010, 93000010, 93000010, 93000010, 93000010, 93000020, 93000020]
- last 10 raw : [150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000, 150000000]
- **no standard time encoding fits this range**


### Sibling files in `market_manipulation_detection/`

(searching for build scripts and metadata sidecars)

```
       9138237 bytes  Documents and Data/data_1115.pkl
         85500 bytes  FinanaceModel/MLP_CNN.py
     462285382 bytes  data/x_data.pkl
         43234 bytes  data/y_dex.pkl
          7622 bytes  sample.csv
```

---

# Permissive raw-data listing (appended)

Generated at 2026-05-04T09:42:45

### `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection/data`

```
kind        size  mtime                path
F      440.87 MB  2023-06-16 16:34:40  x_data.pkl
F       42.22 KB  2023-06-16 16:58:44  y_dex.pkl
```

Extension summary:

```
  .pkl        n=    2  total= 440.91 MB
```

### `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection/Documents and Data`

```
kind        size  mtime                path
F        2.66 MB  2022-12-15 22:01:18  000001.SZ.zip
F        1.74 MB  2022-12-15 22:01:18  Data Introduction .pptx
F        3.01 MB  2022-12-15 22:01:18  Data analytic approach for manipulation detection in stock market.pdf
F        8.71 MB  2022-12-15 22:01:18  data_1115.pkl
```

Extension summary:

```
  .pkl        n=    1  total=   8.71 MB
  .pdf        n=    1  total=   3.01 MB
  .zip        n=    1  total=   2.66 MB
  .pptx       n=    1  total=   1.74 MB
```

### `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection/market_manipulation_detection`

```
kind        size  mtime                path
F        6.00 KB  2026-05-04 09:27:48  .DS_Store
D         0.00 B  2024-03-02 08:13:37  Documents and Data
F        2.66 MB  2022-12-15 22:01:18  Documents and Data/000001.SZ.zip
F        1.74 MB  2022-12-15 22:01:18  Documents and Data/Data Introduction .pptx
F        3.01 MB  2022-12-15 22:01:18  Documents and Data/Data analytic approach for manipulation detection in stock market.pdf
F        8.71 MB  2022-12-15 22:01:18  Documents and Data/data_1115.pkl
D         0.00 B  2024-03-02 08:13:37  FinanaceModel
F       83.50 KB  2023-06-08 17:28:06  FinanaceModel/MLP_CNN.py
F       20.26 MB  2023-06-08 17:27:08  FinanaceModel/wash_graphs_removed.ipynb
F      616.31 KB  2023-10-27 08:02:14  Manipulation.pptx
D         0.00 B  2024-03-02 08:15:03  data
F      440.87 MB  2023-06-16 16:34:40  data/x_data.pkl
F       42.22 KB  2023-06-16 16:58:44  data/y_dex.pkl
F        7.44 KB  2023-10-27 07:52:43  sample.csv
```

Extension summary:

```
  .pkl        n=    3  total= 449.63 MB
  .ipynb      n=    1  total=  20.26 MB
  .pdf        n=    1  total=   3.01 MB
  .zip        n=    1  total=   2.66 MB
  .pptx       n=    2  total=   2.34 MB
  .py         n=    1  total=  83.50 KB
  .csv        n=    1  total=   7.44 KB
  (no-ext)    n=    1  total=   6.00 KB
```

### `/Users/quinference/Library/CloudStorage/Dropbox/Documents/Research/market_manipulation_anomaly_detection`

```
kind        size  mtime                path
F       12.00 KB  2026-05-04 09:27:54  .DS_Store
F      171.96 KB  2023-10-27 08:15:22  Anomaly detection in stock market.pptm
F        1.74 MB  2023-10-27 08:16:33  Data Introduction .pptx
F        2.13 MB  2023-11-24 07:08:34  Manipulation.key
F      409.64 KB  2026-03-19 08:02:13  SD-FMM(revise2)(respond letter).pdf
F        2.31 MB  2026-03-19 08:02:13  SD-FMM(without authors) (revise2)(highlight change).pdf
D         0.00 B  2024-03-02 08:04:00  chinese market manipulaiton cases (sample of 165)
F      473.69 KB  2023-03-12 05:01:30  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书(吴国荣)_中国证券监督管理委员会.pdf
F      728.77 KB  2023-03-12 10:05:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（上海易所试网络信息技术股份有限公司、中泰证券股份有限公司、章源等8名责任人员）_中国证券监督管理委员会.pdf
F      635.53 KB  2023-03-12 10:50:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（上海永邦投资有限公司、朱德洪、杨绍东等4名责任人员）_中国证券监督管理委员会.pdf
F      537.47 KB  2023-03-12 05:26:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（上海通金投资有限公司、刘璟）_中国证券监督管理委员会.pdf
F      657.38 KB  2023-03-12 05:27:36  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（上海阜兴金融控股（集团）有限公司、朱一栋、李卫卫等5名责任人员）_中国证券监督管理委员会.pdf
F      498.19 KB  2023-03-12 10:31:28  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（任良成）_中国证券监督管理委员会.pdf
F      627.93 KB  2023-03-12 09:55:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（何思模）_中国证券监督管理委员会.pdf
F      437.06 KB  2023-03-12 11:25:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（余凯）_中国证券监督管理委员会.pdf
F      480.90 KB  2023-03-12 11:25:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（光大证券股份有限公司、李瑞瑜、水润东）_中国证券监督管理委员会.pdf
F      463.30 KB  2023-03-12 10:03:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（冯志浩）_中国证券监督管理委员会.pdf
F      414.94 KB  2023-03-12 11:03:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（冼锦军）_中国证券监督管理委员会.pdf
F      393.30 KB  2023-03-12 10:45:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘俊峰）_中国证券监督管理委员会.pdf
F      432.66 KB  2023-03-12 05:23:32  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘坚）_中国证券监督管理委员会.pdf
F      479.26 KB  2023-03-12 10:31:58  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘增铖）_中国证券监督管理委员会.pdf
F      455.09 KB  2023-03-12 10:03:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘文金）_中国证券监督管理委员会.pdf
F      593.53 KB  2023-03-12 05:08:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘晓东）_中国证券监督管理委员会.pdf
F      482.53 KB  2023-03-12 05:08:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘晓闽）_中国证券监督管理委员会.pdf
F      556.47 KB  2023-03-12 11:07:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（刘长鸿、冯文渊）_中国证券监督管理委员会.pdf
F      562.82 KB  2023-03-12 05:21:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北京新华汇嘉投资管理有限公司、王卫东）_中国证券监督管理委员会.pdf
F      475.76 KB  2023-03-12 05:24:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北京泛涵投资管理有限公司、陈支左、陈美花）_中国证券监督管理委员会.pdf
F      432.36 KB  2023-03-12 05:24:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北京礼一投资有限公司、深圳礼一投资有限公司、林伟健）_中国证券监督管理委员会.pdf
F      431.66 KB  2023-03-12 11:26:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北京禧达丰证券投资顾问有限公司、白杰旻）_中国证券监督管理委员会.pdf
F      734.42 KB  2023-03-12 11:00:28  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北京艾亿新融资本管理有限公司、张家林、张子恒）_中国证券监督管理委员会.pdf
F      529.29 KB  2023-03-12 10:01:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（北八道集团有限公司、林庆丰、林玉婷等4名责任人员）_中国证券监督管理委员会.pdf
F      415.01 KB  2023-03-12 11:19:10  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（厦门宝拓资源有限公司、陈云卿、苏新）_中国证券监督管理委员会.pdf
F      367.74 KB  2023-03-12 10:49:16  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（叶兆平）_中国证券监督管理委员会.pdf
F      576.87 KB  2023-03-12 05:14:36  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吕乐、陈志龙）_中国证券监督管理委员会.pdf
F      538.00 KB  2023-03-12 05:04:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吕美庆、周学良、黄亮）_中国证券监督管理委员会.pdf
F      491.41 KB  2023-03-12 11:08:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吕美庆）_中国证券监督管理委员会.pdf
F      428.93 KB  2023-03-12 10:07:20  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吴峻乐）_中国证券监督管理委员会.pdf
F      489.41 KB  2023-03-12 05:15:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吴毅健）_中国证券监督管理委员会.pdf
F      674.79 KB  2023-03-12 05:09:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（吴联模）_中国证券监督管理委员会.pdf
F      462.82 KB  2023-03-12 05:04:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（周希俭、陈登科）_中国证券监督管理委员会.pdf
F      356.34 KB  2023-03-12 10:09:08  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（周晨）_中国证券监督管理委员会.pdf
F      712.43 KB  2023-03-12 10:15:42  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（唐汉博、唐园子、袁海林等5名责任人员）_中国证券监督管理委员会.pdf
F      871.66 KB  2023-03-12 10:15:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（唐汉博、王涛）_中国证券监督管理委员会.pdf
F      582.82 KB  2023-03-12 11:09:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（唐汉博）_中国证券监督管理委员会.pdf
F      459.12 KB  2023-03-12 04:44:18  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（唐隆、朱卫）_中国证券监督管理委员会.pdf
F      416.13 KB  2023-03-12 10:46:58  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（唐隆）_中国证券监督管理委员会.pdf
F      542.53 KB  2023-03-12 05:15:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（奔腾集团、张郁达、田永林）_中国证券监督管理委员会.pdf
F      397.27 KB  2023-03-12 05:28:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（姚磊）_中国证券监督管理委员会.pdf
F      467.05 KB  2023-03-12 11:11:12  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（姜为）_中国证券监督管理委员会.pdf
F      452.36 KB  2023-03-12 11:05:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（孙国栋）_中国证券监督管理委员会.pdf
F      395.71 KB  2023-03-12 11:16:18  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（孙德传）_中国证券监督管理委员会.pdf
F      733.85 KB  2023-03-12 05:08:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（孟庆山、杨慧兴）_中国证券监督管理委员会.pdf
F      419.41 KB  2023-03-12 10:03:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（孟祥龙）_中国证券监督管理委员会.pdf
F      446.62 KB  2023-03-12 05:21:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（宁波天一世纪投资有限责任公司、周方洁）_中国证券监督管理委员会.pdf
F      662.53 KB  2023-03-12 04:58:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（宋旭）_中国证券监督管理委员会.pdf
F      426.57 KB  2023-03-12 04:54:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（宗磊、张鹏）_中国证券监督管理委员会.pdf
F      439.58 KB  2023-03-12 04:54:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（宜华集团、刘绍喜等6名责任主体）_中国证券监督管理委员会.pdf
F      445.28 KB  2023-03-12 05:20:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（封建华）_中国证券监督管理委员会.pdf
F      510.11 KB  2023-03-12 10:01:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广州安州投资管理有限公司、王福亮）_中国证券监督管理委员会.pdf
F      564.44 KB  2023-03-12 10:32:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广州市创势翔投资有限公司、黄平、张毅）_中国证券监督管理委员会.pdf
F      539.56 KB  2023-03-12 05:24:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广州市裕鼎投资有限公司、江瑜、胡菊华、吴惠玲）_中国证券监督管理委员会.pdf
F      544.45 KB  2023-03-12 10:15:08  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广州穗富投资管理有限公司、易向军、周岭松等4名责任人员）_中国证券监督管理委员会.pdf
F      515.78 KB  2023-03-12 10:42:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广州穗富投资管理有限公司、易向军、周岭松）_中国证券监督管理委员会.pdf
F      626.51 KB  2023-03-12 05:10:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（广西明利创新实业股份有限公司、林军、唐映等15名责任人员）_中国证券监督管理委员会.pdf
F      569.59 KB  2023-03-12 10:07:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（廖国沛）_中国证券监督管理委员会.pdf
F      442.49 KB  2023-03-12 10:11:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（廖山焱）_中国证券监督管理委员会.pdf
F      707.83 KB  2023-03-12 10:01:42  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（廖英强）_中国证券监督管理委员会.pdf
F      496.88 KB  2023-03-12 05:17:12  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（张平、孙忠泽）_中国证券监督管理委员会.pdf
F      374.57 KB  2023-03-12 11:03:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（张春定）_中国证券监督管理委员会.pdf
F      415.98 KB  2023-03-12 04:56:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（张维）_中国证券监督管理委员会.pdf
F      606.29 KB  2023-03-12 05:14:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（张郁达、张晓敏）_中国证券监督管理委员会.pdf
F      461.87 KB  2023-03-12 05:02:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（张飞、张雄）_中国证券监督管理委员会.pdf
F      391.52 KB  2023-03-12 10:45:30  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（彭旭）_中国证券监督管理委员会.pdf
F      414.63 KB  2023-03-12 10:09:08  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（徐留胜）_中国证券监督管理委员会.pdf
F      575.71 KB  2023-03-12 10:01:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（文高永权、王交英、宋翼湘等4名责任人员）_中国证券监督管理委员会.pdf
F      471.92 KB  2023-03-12 04:43:08  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（易伟）_中国证券监督管理委员会.pdf
F      458.20 KB  2023-03-12 04:58:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（景华）_中国证券监督管理委员会.pdf
F      491.59 KB  2023-03-12 05:13:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（朱康军）_中国证券监督管理委员会.pdf
F      385.10 KB  2023-03-12 10:03:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（朱彬）_中国证券监督管理委员会.pdf
F      447.50 KB  2023-03-12 10:41:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（朱炜明）_中国证券监督管理委员会.pdf
F      471.16 KB  2023-03-12 10:11:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李健）_中国证券监督管理委员会.pdf
F      413.92 KB  2023-03-12 11:09:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李军、张永东）_中国证券监督管理委员会.pdf
F      409.49 KB  2023-03-12 04:54:30  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李勇）_中国证券监督管理委员会.pdf
F      419.42 KB  2023-03-12 04:43:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李卫卫）_中国证券监督管理委员会.pdf
F      430.60 KB  2023-03-12 11:12:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李国东）_中国证券监督管理委员会.pdf
F      400.86 KB  2023-03-12 10:50:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李宁）_中国证券监督管理委员会.pdf
F      495.67 KB  2023-03-12 05:09:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（李直）_中国证券监督管理委员会.pdf
F      636.59 KB  2023-03-12 05:10:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（林军、何忠华、陈志强）_中国证券监督管理委员会.pdf
F      468.62 KB  2023-03-12 04:54:20  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（林永贤、陈宪生、孙磊、徐骏）_中国证券监督管理委员会.pdf
F      392.69 KB  2023-03-12 04:54:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（欧鹏宇）_中国证券监督管理委员会.pdf
F      433.08 KB  2023-03-12 04:58:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江勇）_中国证券监督管理委员会.pdf
F      500.52 KB  2023-03-12 09:55:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江卫东）_中国证券监督管理委员会.pdf
F      398.52 KB  2023-03-12 10:46:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江泉）_中国证券监督管理委员会.pdf
F      688.87 KB  2023-03-12 05:12:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江苏保千里视像科技集团股份有限公司、庄敏、鹿鹏等24名责任人员）_中国证券监督管理委员会.pdf
F      714.75 KB  2023-03-12 10:09:42  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江苏宝利国际投资股份有限公司、周德洪、陈永勤等6名责任人员）_中国证券监督管理委员会.pdf
F      391.01 KB  2023-03-12 04:54:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（江顺平）_中国证券监督管理委员会.pdf
F      481.25 KB  2023-03-12 05:10:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（汪公元）_中国证券监督管理委员会.pdf
F      477.59 KB  2023-03-12 05:24:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（济南华尔泰富投资管理有限公司、时玉祥、田相永）_中国证券监督管理委员会.pdf
F      555.02 KB  2023-03-12 11:17:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（浙江恒逸集团有限公司、楼翔）_中国证券监督管理委员会.pdf
F      582.88 KB  2023-03-12 10:56:18  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（海南亚太实业发展股份有限公司、梁德根、龚成辉等24名责任人员）_中国证券监督管理委员会.pdf
F      592.31 KB  2023-03-12 11:01:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（涂忠华、王伟力、薛文聪等5名责任人员）_中国证券监督管理委员会.pdf
F      483.11 KB  2023-03-12 10:48:08  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（深圳市中鑫富盈基金管理有限公司、李建林、吴峻乐）_中国证券监督管理委员会.pdf
F      522.27 KB  2023-03-12 10:39:12  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（湖北洋丰股份有限公司、杨才学、柴育文等4名责任人员）_中国证券监督管理委员会.pdf
F      428.78 KB  2023-03-12 04:58:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（潘日忠）_中国证券监督管理委员会.pdf
F      534.98 KB  2023-03-12 05:06:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（熊模昌、吴国荣）_中国证券监督管理委员会.pdf
F      712.68 KB  2023-03-12 09:42:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王仕宏、陈杰）_中国证券监督管理委员会.pdf
F      512.52 KB  2023-03-12 05:26:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王凯）_中国证券监督管理委员会.pdf
F      411.42 KB  2023-03-12 11:12:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王华）_中国证券监督管理委员会.pdf
F      455.23 KB  2023-03-12 10:54:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王如增、钟仁志、任斌海）_中国证券监督管理委员会.pdf
F      480.06 KB  2023-03-12 04:45:16  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王宝元）_中国证券监督管理委员会.pdf
F      552.20 KB  2023-03-12 04:56:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王小强）_中国证券监督管理委员会.pdf
F      409.22 KB  2023-03-12 05:20:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王永柯）_中国证券监督管理委员会.pdf
F      443.68 KB  2023-03-12 05:23:32  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王法铜）_中国证券监督管理委员会.pdf
F      384.09 KB  2023-03-12 05:20:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王炎贤）_中国证券监督管理委员会.pdf
F      438.09 KB  2023-03-12 10:12:42  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（王耀沃）_中国证券监督管理委员会.pdf
F      554.72 KB  2023-03-12 10:36:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（现代农装科技股份有限公司、李树君、张海等5名责任人员）_中国证券监督管理委员会.pdf
F      511.46 KB  2023-03-12 05:09:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（田文军）_中国证券监督管理委员会.pdf
F      440.98 KB  2023-03-12 10:31:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（盛巍）_中国证券监督管理委员会.pdf
F      474.41 KB  2023-03-12 04:54:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（盛投和、李佰奇、宋志强）_中国证券监督管理委员会.pdf
F      668.95 KB  2023-03-12 11:07:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（相建康）_中国证券监督管理委员会.pdf
F      405.50 KB  2023-03-12 10:45:32  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（瞿明淑）_中国证券监督管理委员会.pdf
F      577.48 KB  2023-03-12 10:01:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（福建卫东投资集团有限公司、卞友苏、邱一希）_中国证券监督管理委员会.pdf
F      868.88 KB  2023-03-12 05:06:58  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（福建旭诚资产管理有限公司、陈贇、杜闽峰、陈晗、蔡兆艺、林通）_中国证券监督管理委员会.pdf
F      466.85 KB  2023-03-12 05:20:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（福建道冲投资管理有限公司、李盛开、张秋丽）_中国证券监督管理委员会.pdf
F      462.53 KB  2023-03-12 05:09:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（章龙）_中国证券监督管理委员会.pdf
F      466.33 KB  2023-03-12 10:39:10  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（肖海东）_中国证券监督管理委员会.pdf
F      427.02 KB  2023-03-12 10:46:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（胡坤明）_中国证券监督管理委员会.pdf
F      416.88 KB  2023-03-12 11:00:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（胡捷）_中国证券监督管理委员会.pdf
F      418.62 KB  2023-03-12 05:17:14  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（舒逸民）_中国证券监督管理委员会.pdf
F      525.69 KB  2023-03-12 16:27:12  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（苏颜翔）_中国证券监督管理委员会.pdf
F      517.48 KB  2023-03-12 05:21:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（蓝海思通投资控股（上海）有限公司、苏思通）_中国证券监督管理委员会.pdf
F      392.56 KB  2023-03-12 11:25:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（蔡国澍）_中国证券监督管理委员会.pdf
F      487.98 KB  2023-03-12 11:00:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（薛黎明）_中国证券监督管理委员会.pdf
F      650.44 KB  2023-03-12 10:08:26  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（蝶彩资产管理（上海）有限公司、谢风华、阙文斌）_中国证券监督管理委员会.pdf
F      384.87 KB  2023-03-12 11:05:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（袁海林）_中国证券监督管理委员会.pdf
F      492.89 KB  2023-03-12 05:02:22  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（袁渊）_中国证券监督管理委员会.pdf
F      389.46 KB  2023-03-12 11:23:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（袁郑健）_中国证券监督管理委员会.pdf
F      559.77 KB  2023-03-12 05:28:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（褚连江）_中国证券监督管理委员会.pdf
F      413.67 KB  2023-03-12 05:18:16  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（解中力、丁华强）_中国证券监督管理委员会.pdf
F      546.13 KB  2023-03-12 09:42:34  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（谢一峰）_中国证券监督管理委员会.pdf
F      539.97 KB  2023-03-12 05:13:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（赵坚、楼金萍、朱攀峰）_中国证券监督管理委员会.pdf
F      424.29 KB  2023-03-12 10:37:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（赵晨）_中国证券监督管理委员会.pdf
F      461.10 KB  2023-03-12 11:27:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（赵清波、赵波林）_中国证券监督管理委员会.pdf
F      598.38 KB  2023-03-12 05:09:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（邵军、左剑明、胡继峰）_中国证券监督管理委员会.pdf
F      535.32 KB  2023-03-12 05:10:20  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（邹鑫鑫、刘哲）_中国证券监督管理委员会.pdf
F      523.40 KB  2023-03-12 05:28:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（郁红高）_中国证券监督管理委员会.pdf
F      446.12 KB  2023-03-12 04:58:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（郑德胜）_中国证券监督管理委员会.pdf
F      725.61 KB  2023-03-12 05:21:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（郑领滨）_中国证券监督管理委员会.pdf
F      386.59 KB  2023-03-12 11:05:02  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（金建勇）_中国证券监督管理委员会.pdf
F      701.74 KB  2023-03-12 04:58:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（阮浩、嘉和投资、钟山）_中国证券监督管理委员会.pdf
F      500.94 KB  2023-03-12 05:11:36  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈亚发）_中国证券监督管理委员会.pdf
F      431.95 KB  2023-03-12 11:08:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈宏庆）_中国证券监督管理委员会.pdf
F      381.73 KB  2023-03-12 10:46:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈岑宇）_中国证券监督管理委员会.pdf
F      481.15 KB  2023-03-12 05:04:38  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈建铭、谢晶、胡侃）_中国证券监督管理委员会.pdf
F      499.97 KB  2023-03-12 10:58:00  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈明贤）_中国证券监督管理委员会.pdf
F      524.83 KB  2023-03-12 09:55:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈贇）_中国证券监督管理委员会.pdf
F      474.32 KB  2023-03-12 09:55:54  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈贤）_中国证券监督管理委员会.pdf
F      477.04 KB  2023-03-12 04:43:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陈霄）_中国证券监督管理委员会.pdf
F      485.77 KB  2023-03-12 10:58:50  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（陶暘、傅湘南）_中国证券监督管理委员会.pdf
F      436.13 KB  2023-03-12 05:21:46  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（雅利（上海）资产管理有限公司、吕沈强）_中国证券监督管理委员会.pdf
F      483.07 KB  2023-03-12 05:25:28  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（青岛东海恒信投资管理有限公司、史吏、陈建国）_中国证券监督管理委员会.pdf
F      608.26 KB  2023-03-12 04:54:28  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（韩启坤）_中国证券监督管理委员会.pdf
F      367.07 KB  2023-03-12 11:06:40  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（马信琪）_中国证券监督管理委员会.pdf
F      486.41 KB  2023-03-12 10:10:04  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（马永威、曹勇）_中国证券监督管理委员会.pdf
F      427.39 KB  2023-03-12 05:26:10  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（马永威）_中国证券监督管理委员会.pdf
F      587.55 KB  2023-03-12 09:55:52  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（高勇）_中国证券监督管理委员会.pdf
F      401.04 KB  2023-03-12 05:00:06  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（高宗一）_中国证券监督管理委员会.pdf
F      506.78 KB  2023-03-12 05:00:06  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（高鹏）_中国证券监督管理委员会.pdf
F      506.14 KB  2023-03-12 04:54:24  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（鲁成所、王建强）_中国证券监督管理委员会.pdf
F      529.49 KB  2023-03-12 10:12:44  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（鲜言）_中国证券监督管理委员会.pdf
F      475.40 KB  2023-03-12 10:46:56  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（黄信铭、谢冠华、陈囡囡等6名责任人员）_中国证券监督管理委员会.pdf
F      531.26 KB  2023-03-12 04:58:48  chinese market manipulaiton cases (sample of 165)/中国证监会行政处罚决定书（黄鑫、蒋君、徐卫）_中国证券监督管理委员会.pdf
D         0.00 B  2024-03-02 08:04:00  chinese market manipulation cases (sample)
F        6.11 MB  2023-10-27 07:17:10  chinese market manipulation cases (sample)/000001.csv
F        2.29 MB  2023-10-27 07:17:10  chinese market manipulation cases (sample)/1. Analysis of stock market manipulations.pdf
F        6.07 MB  2023-10-27 07:17:10  chinese market manipulation cases (sample)/2. Time Series Contextual Anomaly Detection.pdf
F       95.57 KB  2023-10-27 07:17:10  chinese market manipulation cases (sample)/3. Draft of Meeting.docx
F       52.32 KB  2023-10-27 08:21:28  fig_a.png
D         0.00 B  2025-04-12 16:19:55  market microstructure metrics
F      736.19 KB  2025-04-12 16:18:59  market microstructure metrics/20170215-(002807).csv
F        4.16 MB  2025-04-12 16:19:06  market microstructure metrics/20170405-(002807).csv
F       19.50 KB  2025-04-12 16:18:55  market microstructure metrics/Example Stock(002807).docx
F      771.23 KB  2025-03-05 10:25:30  market microstructure metrics/metrics.ipynb
F      587.88 KB  2025-03-05 10:25:30  market microstructure metrics/metrics.pdf
D         0.00 B  2024-03-02 08:15:03  market_manipulation_detection
F        6.00 KB  2026-05-04 09:27:48  market_manipulation_detection/.DS_Store
D         0.00 B  2024-03-02 08:13:37  market_manipulation_detection/Documents and Data
D         0.00 B  2024-03-02 08:13:37  market_manipulation_detection/FinanaceModel
F      616.31 KB  2023-10-27 08:02:14  market_manipulation_detection/Manipulation.pptx
D         0.00 B  2024-03-02 08:15:03  market_manipulation_detection/data
F        7.44 KB  2023-10-27 07:52:43  market_manipulation_detection/sample.csv
```

Extension summary:

```
  .pdf        n=  170  total=  92.50 MB
  .csv        n=    4  total=  10.99 MB
  .pptx       n=    2  total=   2.34 MB
  .key        n=    1  total=   2.13 MB
  .ipynb      n=    1  total= 771.23 KB
  .pptm       n=    1  total= 171.96 KB
  .docx       n=    2  total= 115.07 KB
  .png        n=    1  total=  52.32 KB
  (no-ext)    n=    2  total=  18.01 KB
```

## Archive contents

### Contents of `000001.SZ.zip`

- archive contains 4 entries
```
     1.97 MB  2022-11-15  000001.SZ/╨╨╟Θ.csv
     9.08 MB  2022-12-15  000001.SZ/╓≡▒╩╬»═╨.csv
     9.96 MB  2022-12-15  000001.SZ/╓≡▒╩│╔╜╗.csv
      0.00 B  2022-12-15  000001.SZ/
```

First 2 KB of `000001.SZ/╨╨╟Θ.csv`:
```
��ô���,����������,��Ȼ��,ʱ��,�ɽ���,�ɽ���,�ɽ���,�ɽ�����,IOPV,�ɽ���־,BS��־,�����ۼƳɽ���,���ճɽ���,��߼�,��ͼ�,���̼�,ǰ����,������1,������2,������3,������4,������5,������6,������7,������8,������9,������10,������1,������2,������3,������4,������5,������6,������7,������8,������9,������10,�����1,�����2,�����3,�����4,�����5,�����6,�����7,�����8,�����9,�����10,������1,������2,������3,������4,������5,������6,������7,������8,������9,������10,��Ȩƽ��������,��Ȩƽ�������,��������,��������,����Ȩָ��,Ʒ������,����Ʒ����,�µ�Ʒ����,��ƽƷ����,
000001.SZ,000001,20221115,91415000,0,0,0,0,0,,,0,0,0,0,0,119500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91500000,0,0,0,0,0,,,0,0,0,0,0,119500,119500,0,0,0,0,0,0,0,0,0,3700,4400,0,0,0,0,0,0,0,0,119500,0,0,0,0,0,0,0,0,0,3700,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91509000,0,0,0,0,0,,,0,0,0,0,0,119500,119100,0,0,0,0,0,0,0,0,0,72100,10100,0,0,0,0,0,0,0,0,119100,0,0,0,0,0,0,0,0,0,72100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91518000,0,0,0,0,0,,,0,0,0,0,0,119500,119100,0,0,0,0,0,0,0,0,0,72700,11600,0,0,0,0,0,0,0,0,119100,0,0,0,0,0,0,0,0,0,72700,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91527000,0,0,0,0,0,,,0,0,0,0,0,119500,119100,0,0,0,0,0,0,0,0,0,78200,6100,0,0,0,0,0,0,0,0,119100,0,0,0,0,0,0,0,0,0,78200,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91536000,0,0,0,0,0,,,0,0,0,0,0,119500,119100,0,0,0,0,0,0,0,0,0,84300,0,0,0,0,0,0,0,0,0,119100,0,0,0,0,0,0,0,0,0,84300,8600,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91545000,0,0,0,0,0,,,0,0,0,0,0,119500,119100,0,0,0,0,0,0,0,0,0,84300,0,0,0,0,0,0,0,0,0,119100,0,0,0,0,0,0,0,0,0,84300,10100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221115,91554000,0,0,0,0,0,,,0,0,0,0,0,119500,119500,0,0,0,0,0,0,0,0,0,124600,5432,0,0,0,0,0,0,0,0,119500,0,0,0,0,0,0,0,0,0,124600,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
000001.SZ,000001,20221
```

