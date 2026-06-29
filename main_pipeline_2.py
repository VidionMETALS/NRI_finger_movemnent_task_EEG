import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert

# =============================================================================
# 1. LOAD DATA
# =============================================================================

eeg = np.load("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/recordings_finger_bhaskar_261025/eeg/finger_movement_20251026_140304.eeg.npy")
ts = np.load("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/recordings_finger_bhaskar_261025/eeg/finger_movement_20251026_140304.ts.npy")

#inspect eeg data and ts data
print("EEG shape:", eeg.shape)
print("TS shape:", ts.shape)

events = pd.read_csv("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/recordings_finger_bhaskar_261025/metadata/finger_movement_20251026_140304_events.csv")

#inspect events file
print(events.head())
print(events.columns)
print(events.shape)

# =============================================================================
# 2. TIME PROCESSING of eeg and ts data
# =============================================================================

ts = ts - ts[0]  # converts absolute time to relative time
fs = 1 / np.mean(np.diff(ts)) # computes sampling frequency
print("Sampling frequency:", fs)
print(ts[:10])

# =============================================================================
# 3. CONVERT TO MNE
# =============================================================================

data_mne = eeg.T * 1e-6  # (channels, samples)
#does two things
# 1. data formatting for mne, transposing from (samples, channels) to (channels, samples)
# 2. scale - 1e-6 converts microvolts to volts why? mne assumes EEG in volts
#without this amplitudes would be 1000000x too large
print(data_mne.shape) # verify if data is transposed

info = mne.create_info(
    ch_names=[f"Ch{i}" for i in range(eeg.shape[1])], 
    sfreq=fs,
    ch_types='eeg'
)
#creates metadata: channel names - ch0, ch1,......, smapling rate, type of data = EEG
#COMBINES SINGLE DATA AND METADATA INTO ONE OBJECT --> CENTRAL OBJECT IN MNE

raw = mne.io.RawArray(data_mne, info) # converted data from numpy to mne eeg object (unprocessed data)
print(raw.ch_names[:10])

#==============================================================================
# 4. MONTAGE RENAMING
#==============================================================================

#load the montage
montage = mne.channels.read_custom_montage("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/recordings_finger_bhaskar_261025/montage144_renamed.sfp")

#get montage channel names
montage_names = montage.ch_names
print("MONTAGE:", montage.ch_names[:10])
#we need these names to align our eeg data channels with the montage channels

#matching channel names in raw and montage files
mapping = { #creates a dictionary like: Ch0 → Cz1, Ch1 → Cz2, ...
    raw.ch_names[i]: montage_names[i] #current names from raw and correct names from montage
    for i in range(len(raw.ch_names)) #assumes same number of channels and same order
} #loop runs for all the channels
#Critical assumption : Channel order in the EEG == Channel order in montage

#renaming the channels
raw.rename_channels(mapping) #renaming channel names - this connects mne to the signal data and electrode positions
print(set(raw.get_channel_types())) #to verify type of data

#apply montage and ignore missing channels
raw.set_montage(montage, on_missing='raise') 
#attaches electrode position to the eeg data

#sainity check 
print(raw.info['chs'][0]['loc'][:3])

#check channels names in raw after renaming 
print("RAW:", raw.ch_names[:10])

#==============================================================================
# 5. PRE-PROCESSING
#==============================================================================

# Downsample
raw.resample(256) #downsampling data and automatically applies anti-alias filter
#reduces brain signalks while keeping brain signals (<100 hz)
print(raw.info['sfreq'])

# Notch filter (50 hz- powerline + 10 hz - harmonic)
raw.notch_filter(freqs=[50], method='iir')
#notch filter harmonics
raw.notch_filter(freqs=[100]) 

#==============================================================================
# 6. COPY OF RAW - TWO BRANCHES 1. PREPROCESSING , 2. BAD CHANNEL DETECION
#==============================================================================

#copy of raw for preprocessing 
raw_process = raw.copy()

#copy for bad channel detection
raw_detect = raw.copy()

#copy for bad channel detection - nearest neighbor 
raw_NN_detect = raw.copy()

# =============================================================================
# 7. BAD CHANNEL DETECTION - MEAN POWER CALCULATIONS
# =============================================================================

# CAR - 1
#raw_detect.set_eeg_reference('average', projection=False) #computes average across all the channels and subtracts from each channel
#print(raw_detect.get_data()[0, :5])
#removes global noise

# Motor band filter (paper: 8–25 Hz)
raw_detect.filter(8, 30)

#extract numpy data
data = raw_detect.get_data()   # shape: (n_channels, n_times)
ch_names = raw_detect.ch_names
print("Shape:", data.shape)

#mean removal (per channel)
channel_means = np.mean(data, axis=1, keepdims=True) #Computes the average value of each channel over time
data = data - channel_means #subtracts each channel's mean from itself 
print(np.mean(data, axis=1)[:5])

#computing power
power = data ** 2 #squares every sample
print(power[0, :5])

#Mean power per channel
mean_power = np.mean(power, axis=1) #averages power over time for each channel
for i in range(min(5, len(mean_power))):
    print(f"{ch_names[i]}: {mean_power[i]:.6e} V^2") #mean power per channels - prints first few channels

#log transform
log_power = np.log(mean_power) #log_power = np.log(np.clip(mean_power, 1e-12, None)) [probable fix if mean power is 0 or negative]
#applies natural logrithm to improve gaussianity
print(f"{ch_names[i]}: {log_power[i]:.4f}")

#z-score trnasformation
z_scores = (log_power - np.mean(log_power)) / np.std(log_power) #standardizes the values
sorted_idx = np.argsort(z_scores)[::-1]  # descending
for i in sorted_idx:
    print(f"{ch_names[i]}: {z_scores[i]:.2f}") # prints z score for all the channels
print("Max z-score:", np.max(z_scores)) #prints max z score value
print("Min z-score:", np.min(z_scores)) #prints min z score value

# Bad channel detection

threshold = 3
bad_idx = np.where(np.abs(z_scores) > threshold)[0]
bad_channels_zscore = [ch_names[i] for i in bad_idx]
print("Bad channels:", bad_channels_zscore)

#==============================================================================
# 8. SPATIAL CORRELATION - NEAREST NEIGHBOR 
#==============================================================================

     #-------------------------------------------------------------------------
     # METRIC 1 - NEAREST NEIGHBOR CORRELATION (NNC)
     #-------------------------------------------------------------------------

# Motor band filter (paper: 8–25 Hz)
raw_NN_detect.filter(8, 30)

#load montage
raw_NN_detect.set_montage(montage)

#get electrodes position
pos = raw_NN_detect.get_montage().get_positions()['ch_pos']

#extract numpy data
data_1 = raw_NN_detect.get_data()   # shape: (n_channels, n_times)
ch_names = raw_NN_detect.ch_names
print("Shape:", data_1.shape)

# Convert to array aligned with channel order
coords = np.array([pos[ch] for ch in ch_names])

#calculate distance matrix
dist_matrix = cdist(coords, coords)  # shape: (n_channels, n_channels)

#finding nearest neighbors
k = 6
neighbors = np.argsort(dist_matrix, axis=1)[:, 1:k+1] # argsort - sorts distances

#compute correlation matrix
corr_matrix = np.corrcoef(data_1)

#data shape
n_channels = data_1.shape[0]

#Compute nearest neighbor correlation
nn_corr = np.zeros(n_channels)

for i in range(n_channels):
    nn_corr[i] = np.mean(corr_matrix[i, neighbors[i]])

#detect bad channel
z_nnc = (nn_corr - np.mean(nn_corr)) / np.std(nn_corr)
bad_idx_c = np.where(z_nnc < -2)[0]
bad_channels_corr = [ch_names[i] for i in bad_idx_c]
print("Bad channels (Correlation):", bad_channels_corr)

# print correlation value for every channel

for ch, corr in zip(ch_names, nn_corr):
    print(f"{ch}: {corr}")

     #-------------------------------------------------------------------------
     # METRIC 2 - NEAREST NEIGHBOR DIFFERENCE 
     #-------------------------------------------------------------------------

#initialize storage
diff_metric = np.zeros(n_channels)

#calculation loop (difference = signal - mean across all channels)
for i in range(n_channels):
    neighbor_signals = data_1[neighbors[i]]   # get signals of 6 neighbors    
    neighbor_mean = np.mean(neighbor_signals, axis=0)  # mean across neighbors
    diff = data_1[i] - neighbor_mean   # difference signal
    diff_metric[i] = np.mean(np.abs(diff))  # converts signal into numbers and then store in diff_metric

#convert values to z - scores
z_diff = (diff_metric - np.mean(diff_metric)) / np.std(diff_metric)
bad_idx_diff = np.where(z_diff > 3)[0]
bad_channels_diff = [ch_names[i] for i in bad_idx_diff]
print("Bad channels (Difference):", bad_channels_diff)

     #-------------------------------------------------------------------------
     # METRIC 3 - CORRELATION - DIFFERENCE (composite score)
     #-------------------------------------------------------------------------

#align direction
z_nnc_inv = -z_nnc # as low NNC = bad --> invert
# NOW - high value = bad channel for both metrics

# composite score - average
composite_score = (z_nnc_inv + z_diff) / 2

#==============================================================================
# 9. BAD CHANNEL DETECTED
#==============================================================================

# bad channel detection

bad_idx_cs = np.where(composite_score > 3)[0]
bad_channels_cs = [ch_names[i] for i in bad_idx_cs]
print("Bad channels (Composite):", bad_channels_cs)

print(np.std(composite_score))
print("Max composite score:", np.max(composite_score)) #prints max z score value
print("Min composite score:", np.min(composite_score)) #prints min z score value

#==============================================================================
# 10. CONTINUE PREPROCESSING 
#==============================================================================

# identified bad channels
# without duplicates, mix of : correlation, difference and composite score
bads = ['CPz48', 'FC3105', 'CP3122']

#band pass filter
raw_process = raw_process.copy().filter(
    l_freq=1.0,
    h_freq=40.0,
    fir_design='firwin'
)

#Bad channel removal or interpolation

# Mark bad channels
raw_process.info['bads'] = ['CPz48', 'FC3105', 'CP3122']

# CAR
raw_process.set_eeg_reference('average', projection=False) #computes average across all the channels and subtracts from each channel
print(raw_process.get_data()[0, :5])
#removes global noise

# PSD plot before ICA
raw_process.plot_psd(fmin=0, fmax=60)

#==============================================================================
# 11. TIME PROCESSING FOR EVENTS DATA AND ALIGNING WITH EEG TO PLOT ICA WITH EVENTS
#==============================================================================

# unmodified version of ts.npy for events time conversion and alignment with eeg
ts_original = np.load("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/recordings_finger_bhaskar_261025/eeg/finger_movement_20251026_140304.ts.npy")

# define columns in events file
time_cols = [
    'prepare_start',
    'raise_start',
    'hold_start',
    'lower_start'
]

# Convert absolute → relative (same reference as EEG)
recording_start = ts_original[0] # define reference (eeg start)

# convert events to relative time
for col in time_cols:
    events[col + '_rel'] = events[col] - recording_start
    # now events are in seconds relative to the eeg

#convert eeg timestamps to relative time    
ts = ts_original - ts_original[0]
# eeg and events are now in the same time reference

# get sampling frequency after (pre-processing)
sfreq = raw_process.info['sfreq'] #ensures events match downsampled eeg 

# define event labels
event_id = {
    'prepare_start': 1,
    'raise_start': 2,
    'hold_start': 3,
    'lower_start': 4
}
print(event_id)

# create an empty array to store events
events_list = []

# convert event time to sampling index
for _, row in events.iterrows(): # loop through each trial
    for col, code in event_id.items(): # loop through each event type
        sample = np.searchsorted(ts_original, row[col]) # convert time to sample index
        events_list.append([sample, 0, code])
        
# compute old and new sampling rate
old_sfreq = 1 / np.mean(np.diff(ts_original))   # gives ~ 1200 
new_sfreq = raw_process.info['sfreq']           # gives ~ 256

# compute scaling ratio
ratio = new_sfreq / old_sfreq # 256 / 1200 ≈ 0.213
#new samples are ~21% of original 

# convert list to numpy array
events_array = np.array(events_list, dtype=int)

#apply scaling of sampling indices
events_array[:, 0] = (events_array[:, 0] * ratio).astype(int)

#==============================================================================
# 12. ICA PIPELINE
#==============================================================================

#configurations of ica model
ica = ICA(
    n_components=40, # should be kept according to the number of channels, we have 144 channels so 40 would be perfect for explicit control or none for full rank
    method='fastica',
    random_state=97,
    max_iter='auto'
)

#actual fitting of ica
ica.fit(raw_process) 

#create epochs for each event (prepare, raise, hold, lower)

# EPOCH FOR PREPARE ===========================================================

epochs_prepare = mne.Epochs(
    raw_process,
    events_array,
    event_id={'prepare': 1},
    tmin=0,
    tmax=5,
    baseline=None,
    preload=True
)

#get ica sources
sources_prepare = ica.get_sources(epochs_prepare)
# now channels --> ICA components
data_prepare = sources_prepare.get_data()
# adjust the time series for the task
times_prepare = epochs_prepare.times

# EPOCH FOR RAISE =============================================================

epochs_raise = mne.Epochs(
    raw_process,
    events_array,
    event_id={'raise': 2},
    tmin=0,
    tmax=2,
    baseline=None,
    preload=True
)

#get ica sources
sources_raise = ica.get_sources(epochs_raise)
# now channels --> ICA components
data_raise = sources_raise.get_data()
# adjust the time series for the task
times_raise = epochs_raise.times

# EPOCH FOR HOLD ==============================================================

epochs_hold = mne.Epochs(
    raw_process,
    events_array,
    event_id={'hold': 3},
    tmin=0,
    tmax=3,
    baseline=None,
    preload=True
)

#get ica sources
sources_hold    = ica.get_sources(epochs_hold)
# now channels --> ICA components
data_hold    = sources_hold.get_data()
# adjust the time series for the task
times_hold = epochs_hold.times

# EPOCH FOR LOWER =============================================================

epochs_lower = mne.Epochs(
    raw_process,
    events_array,
    event_id={'lower': 4},
    tmin=0,
    tmax=2,
    baseline=None,
    preload=True
)

#get ica sources
sources_lower   = ica.get_sources(epochs_lower)
# now channels --> ICA components
data_lower   = sources_lower.get_data()
# adjust the time series for the task
times_lower = epochs_lower.times

# exclude ica components
ica.exclude = [3,4,5,9,13,14,18,24,25,31,39]

# Apply ICA cleaning
clean = ica.apply(raw_process.copy())

#------------------------------------------------------------

# Interpolate instead of dropping (better for spatial consistency)
clean.interpolate_bads(reset_bads=True) 

# Final Motor band filter (paper: 8–25 Hz) - for feature extraction
clean.filter(8, 30, method='iir')

# CAR
clean.set_eeg_reference('average', projection=False) #computes average across all the channels and subtracts from each channel
print(clean.get_data()[0, :5])
#removes global noise

#psd plot
clean.plot_psd(fmin=0, fmax=60)

#------------------------------------------------------------------------------

# SAVING AND RELOADING CLEAN DATA

#plot clean data
clean.plot(scalings='auto', n_channels=20, duration=10) #plot after pre-processing
#save clean data
clean.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean.fif", overwrite=True)

#load clean data
clean = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean.fif", preload=True)

#==============================================================================
# 13. FEATURE EXTRACTION (as per lee et.al paper)
#==============================================================================

# Step 1  - Isolating frequency bands (mu and beta) ---------------------------

# copy of clean for mu band
clean_mu = clean.copy().filter(
    l_freq=8,
    h_freq=13,
    method='iir',
    verbose=False
)
print(clean_mu)

# copy of clean for beta band
clean_beta = clean.copy().filter(
    l_freq=13,
    h_freq=30,
    method='iir',
    verbose=False
)
print(clean_beta)

# Extract numpy data
# converts MNE raw object to Numpy matrix
mu_data = clean_mu.get_data() # mu data
beta_data = clean_beta.get_data() # beta data
print(mu_data.shape)
print(beta_data.shape)

# Step 2 - Compute instantaneous power ----------------------------------------

# squaring each time sample
mu_power = mu_data ** 2 
print(mu_power.shape)
beta_power = beta_data ** 2
print(beta_power.shape)

# Step 3 - Average band power over short durations window (0.25 sec) ----------

# checking sampling rate:
sfreq = clean.info['sfreq']
print(sfreq)

# At freq sample = 256
window_samples = int(0.25 * sfreq) # 64 samples at 256
n_ch, n_times = mu_power.shape
n_windows = n_times // window_samples

# for every time point compute the mean over 64 neighbouring samples
# for mu
mu_power_250ms = (
    mu_power[:, :n_windows * window_samples]
    .reshape(n_ch, n_windows, window_samples)
    .mean(axis=2)
)
print(mu_power_250ms.shape)

# for beta
beta_power_250ms = (
    beta_power[:, :n_windows * window_samples]
    .reshape(n_ch, n_windows, window_samples)
    .mean(axis=2)
)
print(beta_power_250ms.shape)
# this creates non-overlapping windows - not strictly 

# Step 4 - Apply centred moving average (0.75 sec) smoothes nearby points

# for mu 
mu_power_075 = uniform_filter1d(
    mu_power_250ms,
    size=3,
    axis=1,
    mode="nearest"
)
print(mu_power_075.shape)

# for beta
beta_power_075 = uniform_filter1d(
    beta_power_250ms,
    size=3,
    axis=1,
    mode="nearest"
)
print(beta_power_075.shape)

# Step 5 - Log-Transformed ----------------------------------------------------

mu_log = np.log10(mu_power_075 + 1e-12) # for mu 
print(mu_log.shape)
beta_log = np.log10(beta_power_075 + 1e-12) # for beta
print(beta_log.shape)

# Step 6 - 25 sec power shift compensation

# why is it needed? - EEG power slowly changes because of: fatigue, sweating, attention flutuations etc. over minutes.
#                     This is bad because: classifiers might mistake drift for actual motor activity, CSP becomes biased and temporal comparisons become unstable.

# Defining 43 sec window
window_25s = int(25 * 4) # 4 is the new feature_fs after 0.25 smoothening

# moving mean 
mu_baseline = uniform_filter1d(mu_log, size=window_25s, axis=1)
beta_baseline = uniform_filter1d(beta_log, size=window_25s, axis=1)

# Subtract baseline 
mu_comp = mu_log - mu_baseline 
print(mu_comp.shape)
beta_comp = beta_log - beta_baseline
print(beta_comp.shape)

#==============================================================================
# 14. EPOCHING 
#==============================================================================

# CONVERTING CUE TIMINGS INTO FEATURE INDEX 
#EEG is sampled at 256 Hz and cue time is say 51200 samples, cue time = 51200/256 = 200 sec, whereas I no longer have 256 sampling rate I have 4.
pre_windows = int(0.5 / 0.25) # converting pre cue to feature samples 
print(pre_windows)
post_windows = int(12 / 0.25) # converting post cue to feature samples
print(post_windows)
# now my total epoch length includes 51 features

# find the only the prepare events for epoching
prepare_events = events_array[events_array[:,2] == 1]
print(prepare_events.shape)

# convert event locations to feature indices
feature_events = prepare_events[:, 0] // 64 # 64 because window_samples = 0.25 * 256 = 64
print(feature_events.shape)
# now each feature is expressed in the coordinate system of mu_comp

# extracting epochs
# for mu
mu_epochs = []
for ev in feature_events:
    start = ev - pre_windows
    stop  = ev + post_windows + 1
    
    # skip events too close to beginning/end
    if start < 0 or stop > mu_comp.shape[1]:
        continue
    mu_epochs.append(mu_comp[:, start:stop])
mu_epochs = np.array(mu_epochs)
print(mu_epochs.shape)

# for beta
beta_epochs = []
for ev in feature_events:
    start = ev - pre_windows
    stop  = ev + post_windows + 1
    
    # skip events too close to beginning/end
    if start < 0 or stop > beta_comp.shape[1]:
        continue
    beta_epochs.append(beta_comp[:, start:stop])
beta_epochs = np.array(beta_epochs)
print(beta_epochs.shape)

# >>>>>>>>>>>>>>>>>>>>>>>>>> ERDS CURVE/ TOPOGRAPHY >>>>>>>>>>>>>>>>>>>>>>>>>>>

# STEP 1 - Create labels
finger_labels_TP = events["finger"].values

#==============================================================================
# 15. CLASSIFICATION
#==============================================================================

















#==============================================================================
# 16. COPY OF CLEAN DATA FOR STANDALONE CLASSIFICATION MODELS
#==============================================================================

# copy of clean for RF
#clean1_RF = mu_comp.copy()

# copy of clean for SVM
#clean1_SVM = clean1.copy()

# copy of clean for LDA
#clean1_LDA = clean1.copy()

#--------------------------------- RANDOM FOREST ------------------------------







#-------------------------------------- SVM -----------------------------------


















#--------------------------------------- LDA ----------------------------------

#==============================================================================
# COMMON SPATIAL PATTERN (CSP) - DIMENTIONALITY REDUCTION
#==============================================================================

#==============================================================================
# TOPOGRAPHY PLOTTING FOR LDA 
#==============================================================================























#==============================================================================


# Classification
    
    # Machine learning models
        
        # 1. Standalone classifiers 
        
            # 1. LDA
            # 2. RF
            # 3. SVM

        # 2. Ensemble Learning (Stacking)
        
            # Meta-classifier (LDA)
            
        # 3. Hybrid Learning 
        
            # RF for relevant feature extraction
            # SVM or LDA for classification



























