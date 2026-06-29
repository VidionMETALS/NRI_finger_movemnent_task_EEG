import matplotlib.pyplot as plt

# =============================================================================
# VISUALIZATION
# =============================================================================

#copy of raw
raw_before = raw.copy() #SAVE ORIGNAL (IMPORTANT FOR COMPARISON)

#------------------------------------------------------------------------------
# RAW PLOTS
#------------------------------------------------------------------------------
                                                                 
#raw.before plot                                                 
raw_before.plot(n_channels=20, duration=12)                      
                                                                 
#save raw plot                                                   
raw_before.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_before.fif", overwrite=True)
                                                                 
#reopen without running whole code                               
import mne                                                       
raw_before = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_before.fif", preload=True)
raw_before.plot(n_channels=20, duration=12)                      
                                                                 
#------------------------------------------------------------------------------
                                                                 
#raw plot after downsampling and notch                           
raw.plot(n_channels=20, duration=12)                             
                                                                 
#save plot                                                        
raw.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_after-DS_&_NF.fif", overwrite=True)
                                                                 
#reopen without running whole code                               
import mne                                                       
raw = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_after-DS_&_NF.fif", preload=True)
raw.plot(n_channels=20, duration=12)                             
                                                                 
#------------------------------------------------------------------------------
                                                                 
#bad channel detection plot                                      
raw_detect.plot(scalings='auto', n_channels=20, duration=12)     
raw_NN_detect.plot(scalings='auto', n_channels=20, duration=12)     
#save the plot                                              
     
raw_detect.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_detect.fif", overwrite=True)
                                                                 
#reopen without running whole code                               
import mne                                                       
raw_detect = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization/raw_detect.fif", preload=True)
raw_detect.plot(scalings='auto', n_channels=20, duration=12)     
                        
#------------------------------------------------------------------   
                                                            
#pre-processing plot                                             
raw_process.plot(scalings='auto', n_channels=20, duration=12) #plot after pre-processing 
clean.plot(scalings='auto', n_channels=20, duration=10) #plot after pre-processing


                                                                 
#save the plot                                                   
raw_process.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/raw_process_events.fif", overwrite=True)
                                                                 
#reopen without running whole code                               
import mne                                                       
clean = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/raw_clean.fif", preload=True)
clean.plot(scalings='auto', n_channels=20, duration=12)     
                                                                 
#raw.process with events                                         
raw_clean.plot(events=events_array, n_channels=20, duration=5) 

#------------------------------------------------------------------------------
# ASSESSMENT PLOTS
#------------------------------------------------------------------------------

# BAD CHANNEL DETECTION - METHOD 1

#------------------------------------------------------------------------------

#MEAN REMOVAL

plt.figure(figsize=(12,6))

for ch in range(data.shape[0]):
    plt.plot(data[ch], alpha=0.3)

plt.axhline(0, color='black', linestyle='--')
plt.title("All Channels After Mean Removal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#------------------------------------------------------------------------------

#CALCULATING INSTANTANEOUS POWER

#------------------------------------------------------------------------------

# MEAN CHANNEL POWER

# Basic bar plot
# Convert to arrays (just to be safe)
mean_power = np.array(mean_power)
channels = ch_names[:len(mean_power)]

plt.figure(figsize=(12, 5))
plt.bar(channels, mean_power)

plt.xlabel("Channels")
plt.ylabel("Mean Power (V²)")
plt.title("Mean Power per EEG Channel")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------

#log transform

#plot 1
log_power = np.log(mean_power)
channels = ch_names[:len(log_power)]

plt.figure(figsize=(12, 5))
plt.bar(channels, log_power)

plt.xlabel("Channels")
plt.ylabel("Log Mean Power")
plt.title("Log Mean Power per EEG Channel")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# log power gaussian distribution with z score theshold marking

from scipy.stats import norm

# -----------------------------------------------------------------------------
# YOUR DATA
# -----------------------------------------------------------------------------
log_power_gd = np.array(log_power)

# -----------------------------------------------------------------------------
# FIT GAUSSIAN
# -----------------------------------------------------------------------------
mu = np.mean(log_power_gd)
sigma = np.std(log_power_gd)

# -----------------------------------------------------------------------------
# X-axis for Gaussian curve
# -----------------------------------------------------------------------------
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 500)
pdf = norm.pdf(x, mu, sigma)

# -------------------------------
# Z-score thresholds
# -------------------------------
z_levels = [1, 2, 3, 4, 5, 6]

# Convert z -> log power boundaries
bounds = [(mu - z*sigma, mu + z*sigma) for z in z_levels]

# -----------------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

# Histogram
plt.hist(log_power_gd, bins=40, density=True, alpha=0.5, label="Log Power")

# Gaussian curve
plt.plot(x, pdf, 'k', linewidth=2, label="Gaussian fit")

# -----------------------------------------------------------------------------
# Shaded regions (outer to inner)
# -----------------------------------------------------------------------------
colors = ["#ffcccc", "#ffd9b3", "#fff0b3", "#d9ffb3", "#b3e6ff", "#d9d9ff"]

for i, z in enumerate(z_levels[::-1]):  # outermost first
    low, high = mu - z*sigma, mu + z*sigma

    plt.axvspan(low, high, alpha=0.15, color=colors[i],
                label=f"|z| < {z}")

# -----------------------------------------------------------------------------
# Mark mean
# -----------------------------------------------------------------------------
plt.axvline(mu, color='red', linestyle='--', label="Mean")

plt.title("Log Power Distribution with Z-score Threshold Bands")
plt.xlabel("Log Power")
plt.ylabel("Density")
plt.legend()
plt.show()

#------------------------------------------------------------------------------

#z-score plotting

threshold = 6

plt.figure(figsize=(12, 5))
plt.bar(ch_names, z_scores)

plt.axhline(0)
plt.axhline(threshold, linestyle='--')
plt.axhline(-threshold, linestyle='--')

plt.xlabel("Channels")
plt.ylabel("Z-score")
plt.title("Z-scores with Threshold (±6)")

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#==============================================================================

# BAD CHANNEL DETECTION - METHOD 2

#------------------------------------------------------------------------------

# METRIC 1 - CORRELATION

# set threshold
z_thresh = -2.2

plt.figure(figsize=(18,6))

# plot all channels
plt.plot(z_nnc, marker='o', linestyle='-')

# mean line
plt.axhline(0, color='green', linestyle='--', label='Mean (z = 0)')

# threshold line
plt.axhline(z_thresh, color='red', linestyle='--', label=f'z = {z_thresh}')

# highlight bad channels
bad_idx = np.where(z_nnc < z_thresh)[0]
plt.scatter(bad_idx, z_nnc[bad_idx])

# x-axis labels
plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=6)

plt.xlabel("Channels")
plt.ylabel("Z-score (NN Correlation)")
plt.title("Bad Channel Detection (Z-score with Channel Labels)")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# correlation matrix gaussian distribution with z score theshold marking

x = nn_corr

mu = np.mean(x)
sigma = np.std(x)

# -----------------------------------------------------------------------------
# Histogram (empirical distribution)
# -----------------------------------------------------------------------------
plt.figure(figsize=(11, 5))
counts, bins, _ = plt.hist(x, bins=20, density=True, alpha=0.45, label="NN correlation distribution")

# -----------------------------------------------------------------------------
# Gaussian reference curve (smooth)
# -----------------------------------------------------------------------------
z = np.linspace(-6, 6, 1000)
x_pdf = mu + z * sigma
pdf = norm.pdf(z, 0, 1)

plt.plot(x_pdf, pdf, 'r', label="Gaussian reference")

# -----------------------------------------------------------------------------
# Mean line
# -----------------------------------------------------------------------------
plt.axvline(mu, color="black", linewidth=2, label=f"Mean = {mu:.3f}")

# -----------------------------------------------------------------------------
# FIXED BAD CHANNEL RULE: 0.2
# -----------------------------------------------------------------------------
plt.axvline(0.2, color="red", linestyle=":", linewidth=2, label="Bad channel cutoff = 0.2")

# shade bad region (corr < 0.2)
plt.fill_between(
    x_pdf,
    0,
    pdf,
    where=(x_pdf <= 0.2),
    color="red",
    alpha=0.2,
    label="Bad channels (corr < 0.2)"
)

# -----------------------------------------------------------------------------
# Z = -2 reference line
# -----------------------------------------------------------------------------
z2_x = mu - 2 * sigma
plt.axvline(z2_x, color="darkred", linestyle="--", linewidth=2, label="z = -2")

# -----------------------------------------------------------------------------
# Z-score thresholds (-1 to -6)
# -----------------------------------------------------------------------------
for z_val in range(-1, -7, -1):
    x_val = mu + z_val * sigma
    plt.axvline(x_val, linestyle="--", alpha=0.35)

    plt.text(
        x_val,
        max(counts) * 0.85,
        f"z={z_val}",
        rotation=90,
        fontsize=9
    )

# -----------------------------------------------------------------------------
# Axis limits (prevents distortion)
# -----------------------------------------------------------------------------
plt.xlim(mu - 6*sigma, mu + 6*sigma)

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
plt.xlabel("Nearest-neighbor correlation")
plt.ylabel("Density")
plt.title("EEG Channel QC: NN Correlation + Fixed Threshold + Z-scores")
plt.legend()
plt.show()

#------------------------------------------------------------------------------

#METRIC 2 - DIFFERENCE 

# threshold
z_thresh = 2.5

plt.figure(figsize=(18,6))

# plot all channels
plt.plot(z_diff, marker='o', linestyle='-')

# mean line
plt.axhline(0, color='green', linestyle='--', label='Mean (z = 0)')

# threshold line
plt.axhline(z_thresh, color='red', linestyle='--', label='z = +2.5')

# highlight bad channels (high difference = bad)
bad_idx = np.where(z_diff > z_thresh)[0]
plt.scatter(bad_idx, z_diff[bad_idx])

# x-axis labels
plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=6)

plt.xlabel("Channels")
plt.ylabel("Z-score (Difference Metric)")
plt.title("Bad Channel Detection (Difference Metric, z > 2.5)")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# METRIC 3 - COMPOSITE SCORE AND BAD CHANNEL DETECTION

plt.figure(figsize=(14, 5))

z_corr_bad = -z_nnc
z_diff_bad = z_diff

composite_z = (z_corr_bad + z_diff_bad) / 2

# threshold
z_thresh = 3

plt.figure(figsize=(18,6))

# plot composite score
plt.plot(composite_z, marker='o', linestyle='-')

# mean line
plt.axhline(0, color='green', linestyle='--', label='Mean (z = 0)')

# threshold line
plt.axhline(z_thresh, color='red', linestyle='--', label='z = +3')

# highlight bad channels
bad_idx = np.where(composite_z > z_thresh)[0]
plt.scatter(bad_idx, composite_z[bad_idx])

# channel names on x-axis
plt.xticks(range(len(ch_names)), ch_names, rotation=90, fontsize=6)

plt.xlabel("Channels")
plt.ylabel("Composite Z-score")
plt.title("Composite Bad Channel Detection (Threshold = +3)")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#==============================================================================

# BAD CHANNEL INSPECTION PLOT

#bad channels detected
bads = ['CPz48', 'FC3105', 'CP3122', 'CPz39', 'P385']
raw.copy().pick_channels(bads).plot(n_channels=len(bads))

#==============================================================================

# ICA PLOT FOR EVENTS AND COMPONETS WITH PROPERTIES

#plotting components
ica.plot_components()

#------------------------------------------------------------------------------

#plopt ICA properties
picks = range(40) 
ica.plot_properties(raw_process, picks=picks)
plt.show()

#------------------------------------------------------------------------------

#plot time series
ica.plot_sources(raw_process)

# ICA with events


n_components = data_prepare.shape[1]

for ic in range(n_components):

    plt.figure(figsize=(12, 8))

    # ------------------------------ PREPARE ----------------------------------
    plt.subplot(2, 2, 1)
    for ep in range(data_prepare.shape[0]):
        plt.plot(times_prepare, data_prepare[ep, ic], color='gray', alpha=0.2)
    plt.plot(times_prepare, data_prepare[:, ic].mean(axis=0), color='black', linewidth=2)
    plt.title(f"ICA{ic:03d} - prepare")

    # ------------------------------- RAISE -----------------------------------
    plt.subplot(2, 2, 2)
    for ep in range(data_raise.shape[0]):
        plt.plot(times_raise, data_raise[ep, ic], color='gray', alpha=0.2)
    plt.plot(times_raise, data_raise[:, ic].mean(axis=0), color='black', linewidth=2)
    plt.title(f"ICA{ic:03d} - raise")

    # -------------------------------- HOLD -----------------------------------
    plt.subplot(2, 2, 3)
    for ep in range(data_hold.shape[0]):
        plt.plot(times_hold, data_hold[ep, ic], color='gray', alpha=0.2)
    plt.plot(times_hold, data_hold[:, ic].mean(axis=0), color='black', linewidth=2)
    plt.title(f"ICA{ic:03d} - hold")

    # ------------------------------- LOWER -----------------------------------
    plt.subplot(2, 2, 4)
    for ep in range(data_lower.shape[0]):
        plt.plot(times_lower, data_lower[ep, ic], color='gray', alpha=0.2)
    plt.plot(times_lower, data_lower[:, ic].mean(axis=0), color='black', linewidth=2)
    plt.title(f"ICA{ic:03d} - lower")

    plt.suptitle(f"Component {ic}")
    plt.tight_layout()
    plt.show()
    
#------------------------------------------------------------------------------

# exporting ica visualizations to a pdf file

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf_file = "ICA_full_report.pdf"

with PdfPages(pdf_file) as pdf:

    for ic in range(40):

        print(f"Adding ICA{ic:03d} to PDF...")

        # =====================================================================
        # PAGE 1 = ICA Properties
        # =====================================================================
        figs = ica.plot_properties(
            raw_process,
            picks=[ic],
            psd_args={'fmax': 60},
            show=False
        )

        fig_prop = figs[0] if isinstance(figs, list) else figs

        pdf.savefig(fig_prop, bbox_inches='tight')
        plt.close(fig_prop)

        # =====================================================================
        # PAGE 2 = Task Event Responses (ICA space)
        # =====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes = axes.ravel()

        phase_data = {
            "prepare": (data_prepare, times_prepare),
            "raise": (data_raise, times_raise),
            "hold": (data_hold, times_hold),
            "lower": (data_lower, times_lower),
        }

        for ax, (phase, (data, times)) in zip(axes, phase_data.items()):

            ic_signal = data[:, ic, :].mean(axis=0)

            ax.plot(times, ic_signal, color='black', linewidth=1)
            ax.set_title(f'ICA{ic:03d} - {phase}')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (ICA)")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print("Done!")
print(f"Saved PDF: {pdf_file}")

# =============================================================================

# PLOT COMPARISON (BEFORE / AFTER ICA)

import numpy as np
import matplotlib.pyplot as plt
import os

# BEFORE ICA
psd_before = raw_process.compute_psd(
    fmin=0,
    fmax=60,
    verbose=False
)

# AFTER ICA
psd_after = clean.compute_psd(
    fmin=0,
    fmax=60,
    verbose=False
)

# Extract data
freqs = psd_before.freqs

before = psd_before.get_data().mean(axis=0)
after = psd_after.get_data().mean(axis=0)

# Convert to dB
before_db = 10 * np.log10(before)
after_db = 10 * np.log10(after)

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    freqs,
    before_db,
    '--',
    linewidth=2,
    label='Before ICA'
)

plt.plot(
    freqs,
    after_db,
    linewidth=2,
    label='After ICA'
)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('Average PSD Before vs After ICA')
plt.xlim(0,60)

plt.legend()
plt.grid(alpha=0.3)

plt.show()

#==============================================================================

# INTERACTIVE PLOT FOR MU AND BETA DATA 

# mu plot
clean_mu.plot(
    duration=10,       # seconds visible at once
    n_channels=20,     # number shown simultaneously
    scalings='auto',
    title='MU Band (8–13 Hz)',
    block=False
)

#save mu plot
clean_mu.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean_mu.fif", overwrite=True)


# beta plot
clean_beta.plot(
    duration=10,
    n_channels=40,
    scalings='auto',
    title='BETA Band (13–30 Hz)',
    block=False
)

# save beta plot
clean_beta.save("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean_beta.fif", overwrite=True)

#==============================================================================

# instantaneous power plot for mu and beta

# conver numpy values abck to mne object
# for mu
clean_mu_power = mne.io.RawArray(
    mu_power,
    clean_mu.info
)

# for beta
clean_beta_power = mne.io.RawArray(
    beta_power,
    clean_beta.info
)

# plot inst. power for mu
clean_mu_power.plot(
    duration=5,
    n_channels=25,
    scalings='auto',
    decim=4,
    title='MU Instantaneous Power',
    block=False
)

# plot inst. power for beta
clean_beta_power.plot(
    duration=5,
    n_channels=25,
    scalings='auto',
    decim=4,
    title='BETA Instantaneous Power',
    block=False
)

#------------------------------------------------------------------------------

# numeric plot of inst power

times = clean_mu.times  # timestamps in seconds

# MU POWER
plt.figure(figsize=(18, 8))

for ch in range(mu_power.shape[0]):
    plt.plot(times, mu_power[ch], alpha=0.5)

plt.title("Mu Band Instantaneous Power")
plt.xlabel("Time (s)")
plt.ylabel("Power")
plt.grid(True)

plt.show()

# BETA POWER
plt.figure(figsize=(18, 8))

for ch in range(beta_power.shape[0]):
    plt.plot(times, beta_power[ch], alpha=0.5)

plt.title("Beta Band Instantaneous Power")
plt.xlabel("Time (s)")
plt.ylabel("Power")
plt.grid(True)

plt.show()

#==============================================================================

# visualisation log power for mu and beta

# for mu 
times = clean_mu.times

offset = np.std(mu_log) * 5

plt.figure(figsize=(18,12))

for i in range(mu_log.shape[0]):

    plt.plot(
        times,
        mu_log[i] + i * offset,
        linewidth=0.5
    )

plt.xlabel('Time (s)')
plt.ylabel('Channels')
plt.title('MU Log Power — 0.75 s Centered Moving Average')

plt.tight_layout()
plt.show()

# for beta
times = clean_beta.times

offset = np.std(beta_log) * 5

plt.figure(figsize=(18,12))

for i in range(beta_log.shape[0]):

    plt.plot(
        times,
        beta_log[i] + i * offset,
        linewidth=0.5
    )

plt.xlabel('Time (s)')
plt.ylabel('Channels')
plt.title('BETA Log Power — 0.75 s Centered Moving Average')

plt.tight_layout()
plt.show()

#==============================================================================

# MU HISTOGRAM - log power plot


# load clean_mu and clean_beta
clean_mu = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean_mu.fif", preload=True)
clean_beta = mne.io.read_raw_fif("C:/Users/vidyu/Documents/Neuralace/Neuralace research initiative (NRI)/finger-movement-eeg-data/visualization_3_final/clean_beta.fif", preload=True)

plt.figure(figsize=(14, 7))

plt.hist(
    mu_log.flatten(),
    bins=100,
    density=True,
    alpha=0.7
)

plt.xlabel("Mu Log Power")
plt.ylabel("Density")

plt.title("Global Mu Log Power Distribution")

plt.show()

# BETA HISTOGRAM - log power plot

plt.figure(figsize=(14, 7))

plt.hist(
    beta_log.flatten(),
    bins=100,
    density=True,
    alpha=0.7
)

plt.xlabel("Beta Log Power")
plt.ylabel("Density")

plt.title("Global Beta Log Power Distribution")

plt.show()

#==============================================================================

# plotting mu and beta after power shift compensation

# for mu

times = clean_mu.times
offset = np.std(mu_comp) * 5
plt.figure(figsize=(18,12))
yticks = []
ylabels = []
for i in range(mu_comp.shape[0]):

    y = mu_comp[i] + i * offset

    plt.plot(
        times,
        y,
        linewidth=0.5
    )
    yticks.append(i * offset)
    ylabels.append(clean_mu.ch_names[i])

plt.yticks(yticks[::10], ylabels[::10])  # every 10th label
plt.xlabel('Time (s)')
plt.ylabel('Channels')
plt.title('MU Power — Drift Compensated')
plt.tight_layout()
plt.show()

# for beta
times = clean_beta.times
offset = np.std(beta_comp) * 5

plt.figure(figsize=(18,12))

yticks = []
ylabels = []

for i in range(beta_comp.shape[0]):

    y = beta_comp[i] + i * offset

    plt.plot(
        times,
        y,
        linewidth=0.5
    )

    yticks.append(i * offset)
    ylabels.append(clean_beta.ch_names[i])

plt.yticks(yticks[::10], ylabels[::10])  # every 10th label

plt.xlabel('Time (s)')
plt.ylabel('Channels')

plt.title('BETA Power — Drift Compensated')

plt.tight_layout()
plt.show()

#==============================================================================

# MU AND BETA ERD/ERS PLOTS (ALL FINGERS) - RAW

finger_curves = {
    "Thumb":  (thumb_curve,  thumb_curve),
    "Index":  (index_curve,  index_curve),
    "Middle": (middle_curve, middle_curve),
    "Ring":   (ring_curve,   ring_curve),
    "Little": (little_curve, little_curve),
}

for finger, (mu_curve, beta_curve) in finger_curves.items():

    plt.figure(figsize=(12,5))

    plt.plot(
        times,
        mu_curve,
        linewidth=2,
        label='Mu (8-13 Hz)'
    )

    plt.plot(
        times,
        beta_curve,
        linewidth=2,
        label='Beta (13-30 Hz)'
    )

    plt.axhline(
        0,
        color='black',
        linestyle='--'
    )

    plt.axvspan(0,5, alpha=0.1)
    plt.axvspan(5,7, alpha=0.1)
    plt.axvspan(7,10, alpha=0.1)
    plt.axvspan(10,12, alpha=0.1)

    plt.axvline(5,color='gray',linestyle=':')
    plt.axvline(7,color='gray',linestyle=':')
    plt.axvline(10,color='gray',linestyle=':')

    plt.xlabel('Time (s)')
    plt.ylabel('ERD/S')

    plt.title(f'Right {finger}: Mu vs Beta ERD/S')

    plt.legend()
    plt.grid(True)

    plt.show()

#==============================================================================

# smoothening comparison

avg_raw = mu_power.mean(axis=0)

avg_025 = mu_power_avg.mean(axis=0)

avg_075 = mu_power_smooth.mean(axis=0)

plt.figure(figsize=(15,6))

plt.plot(times, avg_raw, alpha=0.4, label='Raw')

plt.plot(times, avg_025, linewidth=2,
         label='0.25 s')

plt.plot(times, avg_075, linewidth=3,
         label='0.75 s')

plt.legend()
plt.grid(True)
plt.show()

#==============================================================================
































