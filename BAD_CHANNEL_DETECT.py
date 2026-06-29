# =========================================================
# BAD CHANNEL DETECTION - MEAN POWER CALCULATIONS
# =========================================================

# BASELINE WINDOW ~36 SEC ---------------------------------

raw_detect_baseline = raw.copy().crop(
    tmin=0,
    tmax=36
)

# CAR - 1
raw_detect_baseline.set_eeg_reference('average', projection=False) #computes average across all the channels and subtracts from each channel
print(raw_detect_baseline.get_data()[0, :5])
#removes global noise

# Motor band filter (paper: 8–25 Hz)
raw_detect_baseline.filter(8, 30)

#extract numpy data
data_baseline = raw_detect_baseline.get_data()
ch_names_baseline = raw_detect_baseline.ch_names
print("Baseline data shape:", data_baseline.shape)

#mean removal (per channel)
channel_means_baseline = np.mean(
    data_baseline,
    axis=1,
    keepdims=True
)
data_baseline = data_baseline - channel_means_baseline
print(
    "Channel means after removal:",
    np.mean(data_baseline, axis=1)[:5]
)

# Compute power
power_baseline = data_baseline ** 2
print(
    "Power samples:",
    power_baseline[0, :5]
)

#Mean power per channel
mean_power_baseline = np.mean(
    power_baseline,
    axis=1
)
for i in range(min(5, len(mean_power_baseline))):
    print(
        f"{ch_names_baseline[i]}: "
        f"{mean_power_baseline[i]:.6e} V^2"
    )
    
#log transform
log_power_baseline = np.log(
    np.clip(mean_power_baseline, 1e-12, None)
) #log_power = np.log(np.clip(mean_power, 1e-12, None)) [probable fix if mean power is 0 or negative]
#applies natural logrithm to improve gaussianity
print(
    f"{ch_names_baseline[0]}: "
    f"{log_power_baseline[0]:.4f}"
)

#z-score trnasformation
z_scores_baseline = (
    log_power_baseline - np.mean(log_power_baseline)
) / np.std(log_power_baseline)
sorted_idx_baseline = np.argsort(
    z_scores_baseline
)[::-1]
for i in sorted_idx_baseline:
    print(
        f"{ch_names_baseline[i]}: "
        f"{z_scores_baseline[i]:.2f}"
    )
print(
    "Max z-score:",
    np.max(z_scores_baseline)
)
print(
    "Min z-score:",
    np.min(z_scores_baseline)
)

# Bad channel detection
threshold_baseline = 3
bad_idx_baseline = np.where(
    np.abs(z_scores_baseline) > threshold_baseline
)[0]
bad_channels_baseline = [
    ch_names_baseline[i]
    for i in bad_idx_baseline
]
print(
    "Bad channels (Baseline):",
    bad_channels_baseline
)

# TASK WINDOW

# ---------------------------------------------
# 1. CROP TASK SEGMENT
# ---------------------------------------------

raw_detect_task = raw.copy().crop(
    tmin=36,
    tmax=raw.times[-1]
)

# ---------------------------------------------
# 2. COMMON AVERAGE REFERENCE (CAR)
# ---------------------------------------------

raw_detect_task.set_eeg_reference(
    'average',
    projection=False
)

print(raw_detect_task.get_data()[0, :5])

# ---------------------------------------------
# 3. MOTOR BAND FILTER
# ---------------------------------------------

raw_detect_task.filter(
    8,
    30
)

# ---------------------------------------------
# 4. EXTRACT NUMPY DATA
# ---------------------------------------------

data_task = raw_detect_task.get_data()

ch_names_task = raw_detect_task.ch_names

print("Task data shape:", data_task.shape)

# ---------------------------------------------
# 5. MEAN REMOVAL (PER CHANNEL)
# ---------------------------------------------

channel_means_task = np.mean(
    data_task,
    axis=1,
    keepdims=True
)

data_task = data_task - channel_means_task

print(
    "Channel means after removal:",
    np.mean(data_task, axis=1)[:5]
)

# ---------------------------------------------
# 6. COMPUTE POWER
# ---------------------------------------------

power_task = data_task ** 2

print(
    "Power samples:",
    power_task[0, :5]
)

# ---------------------------------------------
# 7. MEAN POWER PER CHANNEL
# ---------------------------------------------

mean_power_task = np.mean(
    power_task,
    axis=1
)

for i in range(min(5, len(mean_power_task))):

    print(
        f"{ch_names_task[i]}: "
        f"{mean_power_task[i]:.6e} V^2"
    )

# ---------------------------------------------
# 8. LOG TRANSFORM
# ---------------------------------------------

log_power_task = np.log(
    np.clip(mean_power_task, 1e-12, None)
)

print(
    f"{ch_names_task[0]}: "
    f"{log_power_task[0]:.4f}"
)

# ---------------------------------------------
# 9. Z-SCORE TRANSFORMATION
# ---------------------------------------------

z_scores_task = (
    log_power_task - np.mean(log_power_task)
) / np.std(log_power_task)

sorted_idx_task = np.argsort(
    z_scores_task
)[::-1]

for i in sorted_idx_task:

    print(
        f"{ch_names_task[i]}: "
        f"{z_scores_task[i]:.2f}"
    )

print(
    "Max z-score:",
    np.max(z_scores_task)
)

print(
    "Min z-score:",
    np.min(z_scores_task)
)

# ---------------------------------------------
# 10. BAD CHANNEL DETECTION
# ---------------------------------------------

threshold_task = 3

bad_idx_task = np.where(
    np.abs(z_scores_task) > threshold_task
)[0]

bad_channels_task = [
    ch_names_task[i]
    for i in bad_idx_task
]

print(
    "Bad channels (Task):",
    bad_channels_task
)

#============================================================
# SPATIAL CORRELATION - NEAREST NEIGHBOR 
#============================================================

     #------------------------------------------------------------
     # METRIC 1 - NEAREST NEIGHBOR CORRELATION (NNC)
     #------------------------------------------------------------

            # BASELINE WINDOW -------------------------------------------------

# ---------------------------------------------
# 1. CROP BASELINE SEGMENT
# ---------------------------------------------

raw_NN_detect_baseline = raw.copy().crop(
    tmin=0,
    tmax=36
)

# ---------------------------------------------
# 2. MOTOR BAND FILTER
# ---------------------------------------------

raw_NN_detect_baseline.filter(
    8,
    30
)

# ---------------------------------------------
# 3. APPLY MONTAGE
# ---------------------------------------------

raw_NN_detect_baseline.set_montage(montage)

# ---------------------------------------------
# 4. GET ELECTRODE POSITIONS
# ---------------------------------------------

pos_baseline = (
    raw_NN_detect_baseline
    .get_montage()
    .get_positions()['ch_pos']
)

# ---------------------------------------------
# 5. EXTRACT EEG DATA
# ---------------------------------------------

data_baseline_corr = raw_NN_detect_baseline.get_data()

ch_names_baseline_corr = (
    raw_NN_detect_baseline.ch_names
)

print(
    "Baseline correlation data shape:",
    data_baseline_corr.shape
)

# ---------------------------------------------
# 6. CHANNEL COORDINATES
# ---------------------------------------------

coords_baseline = np.array([
    pos_baseline[ch]
    for ch in ch_names_baseline_corr
])

# ---------------------------------------------
# 7. DISTANCE MATRIX
# ---------------------------------------------

dist_matrix_baseline = cdist(
    coords_baseline,
    coords_baseline
)

# ---------------------------------------------
# 8. FIND NEAREST NEIGHBORS
# ---------------------------------------------

k_baseline = 6

neighbors_baseline = np.argsort(
    dist_matrix_baseline,
    axis=1
)[:, 1:k_baseline+1]

# ---------------------------------------------
# 9. CORRELATION MATRIX
# ---------------------------------------------

corr_matrix_baseline = np.corrcoef(
    data_baseline_corr
)

# ---------------------------------------------
# 10. NUMBER OF CHANNELS
# ---------------------------------------------

n_channels_baseline = (
    data_baseline_corr.shape[0]
)

# ---------------------------------------------
# 11. NEAREST NEIGHBOR CORRELATION
# ---------------------------------------------

nn_corr_baseline = np.zeros(
    n_channels_baseline
)

for i in range(n_channels_baseline):

    nn_corr_baseline[i] = np.mean(
        corr_matrix_baseline[
            i,
            neighbors_baseline[i]
        ]
    )

# ---------------------------------------------
# 12. Z-SCORE TRANSFORMATION
# ---------------------------------------------

z_nnc_baseline = (
    nn_corr_baseline
    - np.mean(nn_corr_baseline)
) / np.std(nn_corr_baseline)

# ---------------------------------------------
# 13. BAD CHANNEL DETECTION
# ---------------------------------------------

corr_threshold_baseline = -2

bad_idx_baseline_corr = np.where(
    z_nnc_baseline < corr_threshold_baseline
)[0]

bad_channels_baseline_corr = [
    ch_names_baseline_corr[i]
    for i in bad_idx_baseline_corr
]

print(
    "Bad channels (Baseline Correlation):",
    bad_channels_baseline_corr
)

# ---------------------------------------------
# 14. PRINT CORRELATION VALUES
# ---------------------------------------------

for ch, corr in zip(
    ch_names_baseline_corr,
    nn_corr_baseline
):

    print(f"{ch}: {corr:.4f}")
    
           # TASK WINDOW ------------------------------------------------------

# ---------------------------------------------
# 1. CROP TASK SEGMENT
# ---------------------------------------------

raw_NN_detect_task = raw.copy().crop(
    tmin=36,
    tmax=raw.times[-1]
)

# ---------------------------------------------
# 2. MOTOR BAND FILTER
# ---------------------------------------------

raw_NN_detect_task.filter(
    8,
    30
)

# ---------------------------------------------
# 3. APPLY MONTAGE
# ---------------------------------------------

raw_NN_detect_task.set_montage(montage)

# ---------------------------------------------
# 4. GET ELECTRODE POSITIONS
# ---------------------------------------------

pos_task = (
    raw_NN_detect_task
    .get_montage()
    .get_positions()['ch_pos']
)

# ---------------------------------------------
# 5. EXTRACT EEG DATA
# ---------------------------------------------

data_task_corr = raw_NN_detect_task.get_data()

ch_names_task_corr = (
    raw_NN_detect_task.ch_names
)

print(
    "Task correlation data shape:",
    data_task_corr.shape
)

# ---------------------------------------------
# 6. CHANNEL COORDINATES
# ---------------------------------------------

coords_task = np.array([
    pos_task[ch]
    for ch in ch_names_task_corr
])

# ---------------------------------------------
# 7. DISTANCE MATRIX
# ---------------------------------------------

dist_matrix_task = cdist(
    coords_task,
    coords_task
)

# ---------------------------------------------
# 8. FIND NEAREST NEIGHBORS
# ---------------------------------------------

k_task = 6

neighbors_task = np.argsort(
    dist_matrix_task,
    axis=1
)[:, 1:k_task+1]

# ---------------------------------------------
# 9. CORRELATION MATRIX
# ---------------------------------------------

corr_matrix_task = np.corrcoef(
    data_task_corr
)

# ---------------------------------------------
# 10. NUMBER OF CHANNELS
# ---------------------------------------------

n_channels_task = (
    data_task_corr.shape[0]
)

# ---------------------------------------------
# 11. NEAREST NEIGHBOR CORRELATION
# ---------------------------------------------

nn_corr_task = np.zeros(
    n_channels_task
)

for i in range(n_channels_task):

    nn_corr_task[i] = np.mean(
        corr_matrix_task[
            i,
            neighbors_task[i]
        ]
    )

# ---------------------------------------------
# 12. Z-SCORE TRANSFORMATION
# ---------------------------------------------

z_nnc_task = (
    nn_corr_task
    - np.mean(nn_corr_task)
) / np.std(nn_corr_task)

# ---------------------------------------------
# 13. BAD CHANNEL DETECTION
# ---------------------------------------------

corr_threshold_task = -2

bad_idx_task_corr = np.where(
    z_nnc_task < corr_threshold_task
)[0]

bad_channels_task_corr = [
    ch_names_task_corr[i]
    for i in bad_idx_task_corr
]

print(
    "Bad channels (Task Correlation):",
    bad_channels_task_corr
)

# ---------------------------------------------
# 14. PRINT CORRELATION VALUES
# ---------------------------------------------

for ch, corr in zip(
    ch_names_task_corr,
    nn_corr_task
):

    print(f"{ch}: {corr:.4f}")

     #----------------------------------------------------------------
     # METRIC 2 - NEAREST NEIGHBOR DIFFERENCE 
     #----------------------------------------------------------------

# BASLINE WINDOW -----------------------------------------------------

# ---------------------------------------------
# 1. INITIALIZE STORAGE
# ---------------------------------------------

diff_metric_baseline = np.zeros(
    n_channels_baseline
)

# ---------------------------------------------
# 2. COMPUTE NEAREST NEIGHBOR DIFFERENCE
# ---------------------------------------------

for i in range(n_channels_baseline):

    # Get signals from nearest neighbors
    neighbor_signals_baseline = (
        data_baseline_corr[
            neighbors_baseline[i]
        ]
    )

    # Mean neighbor signal
    neighbor_mean_baseline = np.mean(
        neighbor_signals_baseline,
        axis=0
    )

    # Difference signal
    diff_baseline = (
        data_baseline_corr[i]
        - neighbor_mean_baseline
    )

    # Mean absolute difference
    diff_metric_baseline[i] = np.mean(
        np.abs(diff_baseline)
    )

# ---------------------------------------------
# 3. Z-SCORE TRANSFORMATION
# ---------------------------------------------

z_diff_baseline = (
    diff_metric_baseline
    - np.mean(diff_metric_baseline)
) / np.std(diff_metric_baseline)

# ---------------------------------------------
# 4. BAD CHANNEL DETECTION
# ---------------------------------------------

diff_threshold_baseline = 3

bad_idx_baseline_diff = np.where(
    z_diff_baseline > diff_threshold_baseline
)[0]

bad_channels_baseline_diff = [
    ch_names_baseline_corr[i]
    for i in bad_idx_baseline_diff
]

print(
    "Bad channels (Baseline Difference):",
    bad_channels_baseline_diff
)

# ---------------------------------------------
# 5. PRINT DIFFERENCE VALUES
# ---------------------------------------------

for ch, diff_val in zip(
    ch_names_baseline_corr,
    diff_metric_baseline
):

    print(f"{ch}: {diff_val:.6e}")

# TASK WINDOW --------------------------------------------------------

# ---------------------------------------------
# 1. INITIALIZE STORAGE
# ---------------------------------------------

diff_metric_task = np.zeros(
    n_channels_task
)

# ---------------------------------------------
# 2. COMPUTE NEAREST NEIGHBOR DIFFERENCE
# ---------------------------------------------

for i in range(n_channels_task):

    # Get signals from nearest neighbors
    neighbor_signals_task = (
        data_task_corr[
            neighbors_task[i]
        ]
    )

    # Mean neighbor signal
    neighbor_mean_task = np.mean(
        neighbor_signals_task,
        axis=0
    )

    # Difference signal
    diff_task = (
        data_task_corr[i]
        - neighbor_mean_task
    )

    # Mean absolute difference
    diff_metric_task[i] = np.mean(
        np.abs(diff_task)
    )

# ---------------------------------------------
# 3. Z-SCORE TRANSFORMATION
# ---------------------------------------------

z_diff_task = (
    diff_metric_task
    - np.mean(diff_metric_task)
) / np.std(diff_metric_task)

# ---------------------------------------------
# 4. BAD CHANNEL DETECTION
# ---------------------------------------------

diff_threshold_task = 3

bad_idx_task_diff = np.where(
    z_diff_task > diff_threshold_task
)[0]

bad_channels_task_diff = [
    ch_names_task_corr[i]
    for i in bad_idx_task_diff
]

print(
    "Bad channels (Task Difference):",
    bad_channels_task_diff
)

# ---------------------------------------------
# 5. PRINT DIFFERENCE VALUES
# ---------------------------------------------

for ch, diff_val in zip(
    ch_names_task_corr,
    diff_metric_task
):

    print(f"{ch}: {diff_val:.6e}")

     #----------------------------------------------------------------
     # METRIC 3 - CORRELATION - DIFFERENCE (composite score)
     #----------------------------------------------------------------

# BASELINE WINDOW ----------------------------------------------------

# ---------------------------------------------
# 1. INVERT CORRELATION Z-SCORES
# ---------------------------------------------

z_nnc_baseline_inv = -z_nnc_baseline

# Low correlation = bad
# After inversion:
# high value = bad channel

# ---------------------------------------------
# 2. COMPUTE COMPOSITE SCORE
# ---------------------------------------------

composite_score_baseline = (
    z_nnc_baseline_inv
    + z_diff_baseline
) / 2

# ---------------------------------------------
# 3. BAD CHANNEL DETECTION
# ---------------------------------------------

composite_threshold_baseline = 6

bad_idx_baseline_composite = np.where(
    composite_score_baseline
    > composite_threshold_baseline
)[0]

bad_channels_baseline_composite = [
    ch_names_baseline_corr[i]
    for i in bad_idx_baseline_composite
]

print(
    "Bad channels (Baseline Composite):",
    bad_channels_baseline_composite
)

# ---------------------------------------------
# 4. PRINT COMPOSITE SCORES
# ---------------------------------------------

for ch, score in zip(
    ch_names_baseline_corr,
    composite_score_baseline
):

    print(f"{ch}: {score:.4f}")

# ---------------------------------------------
# 5. SUMMARY STATISTICS
# ---------------------------------------------

print(
    "Composite score STD:",
    np.std(composite_score_baseline)
)

print(
    "Max composite score:",
    np.max(composite_score_baseline)
)

print(
    "Min composite score:",
    np.min(composite_score_baseline)
)

# TASK WINDOW --------------------------------------------------------

# ---------------------------------------------
# 1. INVERT CORRELATION Z-SCORES
# ---------------------------------------------

z_nnc_task_inv = -z_nnc_task

# Low correlation = bad
# After inversion:
# high value = bad channel

# ---------------------------------------------
# 2. COMPUTE COMPOSITE SCORE
# ---------------------------------------------

composite_score_task = (
    z_nnc_task_inv
    + z_diff_task
) / 2

# ---------------------------------------------
# 3. BAD CHANNEL DETECTION
# ---------------------------------------------

composite_threshold_task = 6

bad_idx_task_composite = np.where(
    composite_score_task
    > composite_threshold_task
)[0]

bad_channels_task_composite = [
    ch_names_task_corr[i]
    for i in bad_idx_task_composite
]

print(
    "Bad channels (Task Composite):",
    bad_channels_task_composite
)

# ---------------------------------------------
# 4. PRINT COMPOSITE SCORES
# ---------------------------------------------

for ch, score in zip(
    ch_names_task_corr,
    composite_score_task
):

    print(f"{ch}: {score:.4f}")

# ---------------------------------------------
# 5. SUMMARY STATISTICS
# ---------------------------------------------

print(
    "Composite score STD:",
    np.std(composite_score_task)
)

print(
    "Max composite score:",
    np.max(composite_score_task)
)

print(
    "Min composite score:",
    np.min(composite_score_task)
)











































