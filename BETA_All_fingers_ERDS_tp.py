
# ====== [ ERDS CALCULATION, PLOTTING CURVE, HEATMAPS AND TOPOGRAPHY ] ====== #

                   # ---------------- BETA --------------- #

#==============================================================================
#---------------------------------- T H U M B ---------------------------------
#==============================================================================

# STEP 2 - marking epochs for Thumb trials only
Thumb_idx = np.where(finger_labels_TP == "Right Thumb")[0]
print("Number of Thumb trials:", len(Thumb_idx))
Thumb_epochs = beta_epochs[Thumb_idx] # selects only Thumb trials
print(Thumb_epochs.shape)

# STEP 3 - extracting Thumb epoch obejcts
data_Thumb = Thumb_epochs # changing variable
print(data_Thumb.shape)

# STEP 4 - Baseline data computation 
baseline_samples = 2 # 2 because in 0.50 only 2 samples are present after feature extraction
baseline = Thumb_epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
print("Baseline shape:", baseline.shape)

# STEP 5 - ERDS computation
erds_Thumb = ((Thumb_epochs - baseline) / np.abs(baseline)) * 100
print(erds_Thumb.shape)

# STEP 6 - Averaging Thumb trials
erds_Thumb_avg = erds_Thumb.mean(axis=0)
print(erds_Thumb_avg.shape)

#------------------------------------------------------------------------------
# ERDS CURVE -                                                    ( T H U M B )
#------------------------------------------------------------------------------

# Average across trials and channels
Thumb_curve = erds_Thumb.mean(axis=(0,1))
print(Thumb_curve.shape)  # should be (n_times,)

# Time vector (0.25 s resolution)
times = np.arange(-0.5, 12.25, 0.25) 
# (start, stop, step) - generates from -0.50 to 12.00 sec, 12.25 is not included
print(times.shape)

plt.figure(figsize=(14,5))

plt.plot(
    times,
    Thumb_curve,
    linewidth=2,
    label="beta ERDS"
)

# Zero line
plt.axhline(0, color='green', linestyle='--')

# Cue and phase boundaries
plt.axvline(0,  color='red',   linestyle='--', label='Cue')
plt.axvline(5,  color='black', linestyle='--')
plt.axvline(7,  color='black', linestyle='--')
plt.axvline(10, color='black', linestyle='--')

# Phase shading
plt.axvspan(-0.5, 0, alpha=0.15)
plt.axvspan(0, 5, alpha=0.10)
plt.axvspan(5, 7, alpha=0.10)
plt.axvspan(7, 10, alpha=0.10)
plt.axvspan(10, 12, alpha=0.10)

# Place labels above the curve
y_top = np.max(Thumb_curve) * 1.05

plt.text(-0.25, y_top, "Baseline", ha='center')
plt.text(2.5,  y_top, "Prepare", ha='center')
plt.text(6.0,  y_top, "Raise", ha='center')
plt.text(8.5,  y_top, "Hold", ha='center')
plt.text(11.0, y_top, "Lower", ha='center')

plt.xlabel("Time (s)")
plt.ylabel("ERDS (%)")
plt.title("Right Thumb μ-band Global ERDS")

plt.xlim([-0.5, 12])

plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------------------------------------------
# ERDS CURVE - PHASE STRUCTURED (SHADED)                          ( T H U M B )
#------------------------------------------------------------------------------

plt.figure(figsize=(14, 5))

# 1. Plot the continuous average beta ERDS curve line
plt.plot(times, Thumb_curve, color='black', linewidth=2, zorder=3)

# 2. Add the baseline 0.0 line reference
plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)

# 3. Apply custom dual-color fill underneath the curve dynamically
# Fill positive values (ERS) in soft red
plt.fill_between(times, Thumb_curve, 0, where=(Thumb_curve >= 0), 
                 color='#f4c7c7', alpha=0.8, label='ERS')

# Fill negative values (ERD) in soft blue
plt.fill_between(times, Thumb_curve, 0, where=(Thumb_curve < 0), 
                 color='#c7cee8', alpha=0.8, label='ERD')

# 4. Define the precise time boundaries for your 7 distinct phases
phase_boundaries = [-1.0, 0.0, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.0]
phase_labels = ["Baseline", "Prepare", "Prep-Raise", "Raise", "Raise-Hold", "Hold", "Hold-Lower", "Lower"]

# 5. Draw dashed vertical segment dividers and alternative background shading
for idx in range(len(phase_boundaries) - 1):
    start = phase_boundaries[idx]
    end = phase_boundaries[idx + 1]
    
    # Draw vertical bounding line
    plt.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Apply subtle alternating gray striping background for crisp scannability
    if idx % 2 == 0:
        plt.axvspan(start, end, color='#f5f5f5', alpha=0.5, zorder=0)
    else:
        plt.axvspan(start, end, color='#ffffff', alpha=1.0, zorder=0)
        
    # Calculate the midpoints to centered-align labels along the top axis ceiling
    x_text_pos = start + (end - start) / 2
    y_text_pos = np.max(Thumb_curve) * 0.85
    plt.text(x_text_pos, y_text_pos, phase_labels[idx], 
             ha='center', va='center', fontsize=9, fontweight='semibold', color='#333333')

# Final vertical line at the layout boundary
plt.axvline(phase_boundaries[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

# 6. Styling adjustments
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('ERD / ERS', fontsize=11, fontweight='bold')
plt.title('beta ERDS (Phase Structured) - Right Thumb', fontsize=13, fontweight='bold', pad=15)

# Tighten limits around your epoch bounds
plt.xlim([-1.0, 12.0])

# Clean layout with upper-right corner legend configuration
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.grid(False) # Turn off standard grid grids to keep phase shading clear

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# TOPOGRAPHY HEATMAPS -                                           ( T H U M B )
#------------------------------------------------------------------------------

# <><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><><><><><><><>

#==============================================================================
#---------------------------------- I N D E X ---------------------------------
#==============================================================================

# STEP 2 - marking epochs for Index trials only
Index_idx = np.where(finger_labels_TP == "Right Index")[0]
print("Number of Index trials:", len(Index_idx))
Index_epochs = beta_epochs[Index_idx] # selects only Index trials
print(Index_epochs.shape)

# STEP 3 - extracting Index epoch obejcts
data_Index = Index_epochs # changing variable
print(data_Index.shape)

# STEP 4 - Baseline data computation 
baseline_samples = 2 # 2 because in 0.50 only 2 samples are present after feature extraction
baseline = Index_epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
print("Baseline shape:", baseline.shape)

# STEP 5 - ERDS computation
erds_Index = ((Index_epochs - baseline) / np.abs(baseline)) * 100
print(erds_Index.shape)

# STEP 6 - Averaging Index trials
erds_Index_avg = erds_Index.mean(axis=0)
print(erds_Index_avg.shape)

#------------------------------------------------------------------------------
# ERDS CURVE -                                                    ( I N D E X )
#------------------------------------------------------------------------------

# Average across trials and channels
Index_curve = erds_Index.mean(axis=(0,1))
print(Index_curve.shape)  # should be (n_times,)

# Time vector (0.25 s resolution)
times = np.arange(-0.5, 12.25, 0.25) # (start, stop, step) - generates from -0.50 to 12.00 sec, 12.25 is not included
print(times.shape)

plt.figure(figsize=(14,5))

plt.plot(
    times,
    Index_curve,
    linewidth=2,
    label="beta ERDS"
)

# Zero line
plt.axhline(0, color='green', linestyle='--')

# Cue and phase boundaries
plt.axvline(0,  color='red',   linestyle='--', label='Cue')
plt.axvline(5,  color='black', linestyle='--')
plt.axvline(7,  color='black', linestyle='--')
plt.axvline(10, color='black', linestyle='--')

# Phase shading
plt.axvspan(-0.5, 0, alpha=0.15)
plt.axvspan(0, 5, alpha=0.10)
plt.axvspan(5, 7, alpha=0.10)
plt.axvspan(7, 10, alpha=0.10)
plt.axvspan(10, 12, alpha=0.10)

# Place labels above the curve
y_top = np.max(Index_curve) * 1.05

plt.text(-0.25, y_top, "Baseline", ha='center')
plt.text(2.5,  y_top, "Prepare", ha='center')
plt.text(6.0,  y_top, "Raise", ha='center')
plt.text(8.5,  y_top, "Hold", ha='center')
plt.text(11.0, y_top, "Lower", ha='center')

plt.xlabel("Time (s)")
plt.ylabel("ERDS (%)")
plt.title("Right Index μ-band Global ERDS")

plt.xlim([-0.5, 12])

plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------------------------------------------
# ERDS CURVE - PHASE STRUCTURED (SHADED)                          ( I N D E X )
#------------------------------------------------------------------------------

plt.figure(figsize=(14, 5))

# 1. Plot the continuous average beta ERDS curve line
plt.plot(times, Index_curve, color='black', linewidth=2, zorder=3)

# 2. Add the baseline 0.0 line reference
plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)

# 3. Apply custom dual-color fill underneath the curve dynamically
# Fill positive values (ERS) in soft red
plt.fill_between(times, Index_curve, 0, where=(Index_curve >= 0), 
                 color='#f4c7c7', alpha=0.8, label='ERS')

# Fill negative values (ERD) in soft blue
plt.fill_between(times, Index_curve, 0, where=(Index_curve < 0), 
                 color='#c7cee8', alpha=0.8, label='ERD')

# 4. Define the precise time boundaries for your 7 distinct phases
phase_boundaries = [-1.0, 0.0, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.0]
phase_labels = ["Baseline", "Prepare", "Prep-Raise", "Raise", "Raise-Hold", "Hold", "Hold-Lower", "Lower"]

# 5. Draw dashed vertical segment dividers and alternative background shading
for idx in range(len(phase_boundaries) - 1):
    start = phase_boundaries[idx]
    end = phase_boundaries[idx + 1]
    
    # Draw vertical bounding line
    plt.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Apply subtle alternating gray striping background for crisp scannability
    if idx % 2 == 0:
        plt.axvspan(start, end, color='#f5f5f5', alpha=0.5, zorder=0)
    else:
        plt.axvspan(start, end, color='#ffffff', alpha=1.0, zorder=0)
        
    # Calculate the midpoints to centered-align labels along the top axis ceiling
    x_text_pos = start + (end - start) / 2
    y_text_pos = np.max(Index_curve) * 0.85
    plt.text(x_text_pos, y_text_pos, phase_labels[idx], 
             ha='center', va='center', fontsize=9, fontweight='semibold', color='#333333')

# Final vertical line at the layout boundary
plt.axvline(phase_boundaries[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

# 6. Styling adjustments
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('ERD / ERS', fontsize=11, fontweight='bold')
plt.title('beta ERDS (Phase Structured) - Right Index', fontsize=13, fontweight='bold', pad=15)

# Tighten limits around your epoch bounds
plt.xlim([-1.0, 12.0])

# Clean layout with upper-right corner legend configuration
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.grid(False) # Turn off standard grid grids to keep phase shading clear

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# TOPOGRAPHY HEATMAPS -                                           ( I N D E X )
#------------------------------------------------------------------------------

# <><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><><><><><><><>

#==============================================================================
#---------------------------------- M I D D L E -------------------------------
#==============================================================================

# STEP 2 - marking epochs for Middle trials only
Middle_idx = np.where(finger_labels_TP == "Right Middle")[0]
print("Number of Middle trials:", len(Middle_idx))
Middle_epochs = beta_epochs[Middle_idx] # selects only Middle trials
print(Middle_epochs.shape)

# STEP 3 - extracting Middle epoch obejcts
data_Middle = Middle_epochs # changing variable
print(data_Middle.shape)

# STEP 4 - Baseline data computation 
baseline_samples = 2 # 2 because in 0.50 only 2 samples are present after feature extraction
baseline = Middle_epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
print("Baseline shape:", baseline.shape)

# STEP 5 - ERDS computation
erds_Middle = ((Middle_epochs - baseline) / np.abs(baseline)) * 100
print(erds_Middle.shape)

# STEP 6 - Averaging Middle trials
erds_Middle_avg = erds_Middle.mean(axis=0)
print(erds_Middle_avg.shape)

#------------------------------------------------------------------------------
# ERDS CURVE -                                                  ( M I D D L E )
#------------------------------------------------------------------------------

# Average across trials and channels
Middle_curve = erds_Middle.mean(axis=(0,1))
print(Middle_curve.shape)  # should be (n_times,)

# Time vector (0.25 s resolution)
times = np.arange(-0.5, 12.25, 0.25) # (start, stop, step) - generates from -0.50 to 12.00 sec, 12.25 is not included
print(times.shape)

plt.figure(figsize=(14,5))

plt.plot(
    times,
    Middle_curve,
    linewidth=2,
    label="beta ERDS"
)

# Zero line
plt.axhline(0, color='green', linestyle='--')

# Cue and phase boundaries
plt.axvline(0,  color='red',   linestyle='--', label='Cue')
plt.axvline(5,  color='black', linestyle='--')
plt.axvline(7,  color='black', linestyle='--')
plt.axvline(10, color='black', linestyle='--')

# Phase shading
plt.axvspan(-0.5, 0, alpha=0.15)
plt.axvspan(0, 5, alpha=0.10)
plt.axvspan(5, 7, alpha=0.10)
plt.axvspan(7, 10, alpha=0.10)
plt.axvspan(10, 12, alpha=0.10)

# Place labels above the curve
y_top = np.max(Middle_curve) * 1.05

plt.text(-0.25, y_top, "Baseline", ha='center')
plt.text(2.5,  y_top, "Prepare", ha='center')
plt.text(6.0,  y_top, "Raise", ha='center')
plt.text(8.5,  y_top, "Hold", ha='center')
plt.text(11.0, y_top, "Lower", ha='center')

plt.xlabel("Time (s)")
plt.ylabel("ERDS (%)")
plt.title("Right Middle μ-band Global ERDS")

plt.xlim([-0.5, 12])

plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------------------------------------------
# ERDS CURVE - PHASE STRUCTURED (SHADED)                        ( M I D D L E )
#------------------------------------------------------------------------------

plt.figure(figsize=(14, 5))

# 1. Plot the continuous average beta ERDS curve line
plt.plot(times, Middle_curve, color='black', linewidth=2, zorder=3)

# 2. Add the baseline 0.0 line reference
plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)

# 3. Apply custom dual-color fill underneath the curve dynamically
# Fill positive values (ERS) in soft red
plt.fill_between(times, Middle_curve, 0, where=(Middle_curve >= 0), 
                 color='#f4c7c7', alpha=0.8, label='ERS')

# Fill negative values (ERD) in soft blue
plt.fill_between(times, Middle_curve, 0, where=(Middle_curve < 0), 
                 color='#c7cee8', alpha=0.8, label='ERD')

# 4. Define the precise time boundaries for your 7 distinct phases
phase_boundaries = [-1.0, 0.0, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.0]
phase_labels = ["Baseline", "Prepare", "Prep-Raise", "Raise", "Raise-Hold", "Hold", "Hold-Lower", "Lower"]

# 5. Draw dashed vertical segment dividers and alternative background shading
for idx in range(len(phase_boundaries) - 1):
    start = phase_boundaries[idx]
    end = phase_boundaries[idx + 1]
    
    # Draw vertical bounding line
    plt.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Apply subtle alternating gray striping background for crisp scannability
    if idx % 2 == 0:
        plt.axvspan(start, end, color='#f5f5f5', alpha=0.5, zorder=0)
    else:
        plt.axvspan(start, end, color='#ffffff', alpha=1.0, zorder=0)
        
    # Calculate the midpoints to centered-align labels along the top axis ceiling
    x_text_pos = start + (end - start) / 2
    y_text_pos = np.max(Middle_curve) * 0.85
    plt.text(x_text_pos, y_text_pos, phase_labels[idx], 
             ha='center', va='center', fontsize=9, fontweight='semibold', color='#333333')

# Final vertical line at the layout boundary
plt.axvline(phase_boundaries[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

# 6. Styling adjustments
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('ERD / ERS', fontsize=11, fontweight='bold')
plt.title('beta ERDS (Phase Structured) - Right Middle', fontsize=13, fontweight='bold', pad=15)

# Tighten limits around your epoch bounds
plt.xlim([-1.0, 12.0])

# Clean layout with upper-right corner legend configuration
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.grid(False) # Turn off standard grid grids to keep phase shading clear

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# TOPOGRAPHY HEATMAPS -                                         ( M I D D L E )
#------------------------------------------------------------------------------

# <><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><><><><><><><>

#==============================================================================
#---------------------------------- R I N G -----------------------------------
#==============================================================================

# STEP 2 - marking epochs for Ring trials only
Ring_idx = np.where(finger_labels_TP == "Right Ring")[0]
print("Number of Ring trials:", len(Ring_idx))
Ring_epochs = beta_epochs[Ring_idx] # selects only Ring trials
print(Ring_epochs.shape)

# STEP 3 - extracting Ring epoch obejcts
data_Ring = Ring_epochs # changing variable
print(data_Ring.shape)

# STEP 4 - Baseline data computation 
baseline_samples = 2 # 2 because in 0.50 only 2 samples are present after feature extraction
baseline = Ring_epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
print("Baseline shape:", baseline.shape)

# STEP 5 - ERDS computation
erds_Ring = ((Ring_epochs - baseline) / np.abs(baseline)) * 100
print(erds_Ring.shape)

# STEP 6 - Averaging Ring trials
erds_Ring_avg = erds_Ring.mean(axis=0)
print(erds_Ring_avg.shape)

#------------------------------------------------------------------------------
# ERDS CURVE -                                                      ( R I N G )
#------------------------------------------------------------------------------

# Average across trials and channels
Ring_curve = erds_Ring.mean(axis=(0,1))
print(Ring_curve.shape)  # should be (n_times,)

# Time vector (0.25 s resolution)
times = np.arange(-0.5, 12.25, 0.25) # (start, stop, step) - generates from -0.50 to 12.00 sec, 12.25 is not included
print(times.shape)

plt.figure(figsize=(14,5))

plt.plot(
    times,
    Ring_curve,
    linewidth=2,
    label="beta ERDS"
)

# Zero line
plt.axhline(0, color='green', linestyle='--')

# Cue and phase boundaries
plt.axvline(0,  color='red',   linestyle='--', label='Cue')
plt.axvline(5,  color='black', linestyle='--')
plt.axvline(7,  color='black', linestyle='--')
plt.axvline(10, color='black', linestyle='--')

# Phase shading
plt.axvspan(-0.5, 0, alpha=0.15)
plt.axvspan(0, 5, alpha=0.10)
plt.axvspan(5, 7, alpha=0.10)
plt.axvspan(7, 10, alpha=0.10)
plt.axvspan(10, 12, alpha=0.10)

# Place labels above the curve
y_top = np.max(Ring_curve) * 1.05

plt.text(-0.25, y_top, "Baseline", ha='center')
plt.text(2.5,  y_top, "Prepare", ha='center')
plt.text(6.0,  y_top, "Raise", ha='center')
plt.text(8.5,  y_top, "Hold", ha='center')
plt.text(11.0, y_top, "Lower", ha='center')

plt.xlabel("Time (s)")
plt.ylabel("ERDS (%)")
plt.title("Right Ring μ-band Global ERDS")

plt.xlim([-0.5, 12])

plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------------------------------------------
# ERDS CURVE - PHASE STRUCTURED (SHADED)                            ( R I N G )
#------------------------------------------------------------------------------

plt.figure(figsize=(14, 5))

# 1. Plot the continuous average beta ERDS curve line
plt.plot(times, Ring_curve, color='black', linewidth=2, zorder=3)

# 2. Add the baseline 0.0 line reference
plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)

# 3. Apply custom dual-color fill underneath the curve dynamically
# Fill positive values (ERS) in soft red
plt.fill_between(times, Ring_curve, 0, where=(Ring_curve >= 0), 
                 color='#f4c7c7', alpha=0.8, label='ERS')

# Fill negative values (ERD) in soft blue
plt.fill_between(times, Ring_curve, 0, where=(Ring_curve < 0), 
                 color='#c7cee8', alpha=0.8, label='ERD')

# 4. Define the precise time boundaries for your 7 distinct phases
phase_boundaries = [-1.0, 0.0, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.0]
phase_labels = ["Baseline", "Prepare", "Prep-Raise", "Raise", "Raise-Hold", "Hold", "Hold-Lower", "Lower"]

# 5. Draw dashed vertical segment dividers and alternative background shading
for idx in range(len(phase_boundaries) - 1):
    start = phase_boundaries[idx]
    end = phase_boundaries[idx + 1]
    
    # Draw vertical bounding line
    plt.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Apply subtle alternating gray striping background for crisp scannability
    if idx % 2 == 0:
        plt.axvspan(start, end, color='#f5f5f5', alpha=0.5, zorder=0)
    else:
        plt.axvspan(start, end, color='#ffffff', alpha=1.0, zorder=0)
        
    # Calculate the midpoints to centered-align labels along the top axis ceiling
    x_text_pos = start + (end - start) / 2
    y_text_pos = np.max(Ring_curve) * 0.85
    plt.text(x_text_pos, y_text_pos, phase_labels[idx], 
             ha='center', va='center', fontsize=9, fontweight='semibold', color='#333333')

# Final vertical line at the layout boundary
plt.axvline(phase_boundaries[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

# 6. Styling adjustments
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('ERD / ERS', fontsize=11, fontweight='bold')
plt.title('beta ERDS (Phase Structured) - Right Ring', fontsize=13, fontweight='bold', pad=15)

# Tighten limits around your epoch bounds
plt.xlim([-1.0, 12.0])

# Clean layout with upper-right corner legend configuration
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.grid(False) # Turn off standard grid grids to keep phase shading clear

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# TOPOGRAPHY HEATMAPS -                                             ( R I N G )
#------------------------------------------------------------------------------

# <><><><><><><><><><><><><><><><><><><><>><><><><><><><><><><><><><><><><><><>

#==============================================================================
#---------------------------------- L I T T L E -------------------------------
#==============================================================================

# STEP 2 - marking epochs for Little trials only
Little_idx = np.where(finger_labels_TP == "Right Little")[0]
print("Number of Little trials:", len(Little_idx))
Little_epochs = beta_epochs[Little_idx] # selects only Little trials
print(Little_epochs.shape)

# STEP 3 - extracting Little epoch obejcts
data_Little = Little_epochs # changing variable
print(data_Little.shape)

# STEP 4 - Baseline data computation 
baseline_samples = 2 # 2 because in 0.50 only 2 samples are present after feature extraction
baseline = Little_epochs[:, :, :baseline_samples].mean(axis=2, keepdims=True)
print("Baseline shape:", baseline.shape)

# STEP 5 - ERDS computation
erds_Little = ((Little_epochs - baseline) / np.abs(baseline)) * 100
print(erds_Little.shape)

# STEP 6 - Averaging Little trials
erds_Little_avg = erds_Little.mean(axis=0)
print(erds_Little_avg.shape)

#------------------------------------------------------------------------------
# ERDS CURVE -                                                  ( L I T T L E )
#------------------------------------------------------------------------------

# Average across trials and channels
Little_curve = erds_Little.mean(axis=(0,1))
print(Little_curve.shape)  # should be (n_times,)

# Time vector (0.25 s resolution)
times = np.arange(-0.5, 12.25, 0.25) # (start, stop, step) - generates from -0.50 to 12.00 sec, 12.25 is not included
print(times.shape)

plt.figure(figsize=(14,5))

plt.plot(
    times,
    Little_curve,
    linewidth=2,
    label="beta ERDS"
)

# Zero line
plt.axhline(0, color='green', linestyle='--')

# Cue and phase boundaries
plt.axvline(0,  color='red',   linestyle='--', label='Cue')
plt.axvline(5,  color='black', linestyle='--')
plt.axvline(7,  color='black', linestyle='--')
plt.axvline(10, color='black', linestyle='--')

# Phase shading
plt.axvspan(-0.5, 0, alpha=0.15)
plt.axvspan(0, 5, alpha=0.10)
plt.axvspan(5, 7, alpha=0.10)
plt.axvspan(7, 10, alpha=0.10)
plt.axvspan(10, 12, alpha=0.10)

# Place labels above the curve
y_top = np.max(Little_curve) * 1.05

plt.text(-0.25, y_top, "Baseline", ha='center')
plt.text(2.5,  y_top, "Prepare", ha='center')
plt.text(6.0,  y_top, "Raise", ha='center')
plt.text(8.5,  y_top, "Hold", ha='center')
plt.text(11.0, y_top, "Lower", ha='center')

plt.xlabel("Time (s)")
plt.ylabel("ERDS (%)")
plt.title("Right Little μ-band Global ERDS")

plt.xlim([-0.5, 12])

plt.grid(True)
plt.legend()

plt.show()

#------------------------------------------------------------------------------
# ERDS CURVE - PHASE STRUCTURED (SHADED)                        ( L I T T L E )
#------------------------------------------------------------------------------

plt.figure(figsize=(14, 5))

# 1. Plot the continuous average beta ERDS curve line
plt.plot(times, Little_curve, color='black', linewidth=2, zorder=3)

# 2. Add the baseline 0.0 line reference
plt.axhline(0, color='black', linestyle='-', linewidth=1, zorder=2)

# 3. Apply custom dual-color fill underneath the curve dynamically
# Fill positive values (ERS) in soft red
plt.fill_between(times, Little_curve, 0, where=(Little_curve >= 0), 
                 color='#f4c7c7', alpha=0.8, label='ERS')

# Fill negative values (ERD) in soft blue
plt.fill_between(times, Little_curve, 0, where=(Little_curve < 0), 
                 color='#c7cee8', alpha=0.8, label='ERD')

# 4. Define the precise time boundaries for your 7 distinct phases
phase_boundaries = [-1.0, 0.0, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 12.0]
phase_labels = ["Baseline", "Prepare", "Prep-Raise", "Raise", "Raise-Hold", "Hold", "Hold-Lower", "Lower"]

# 5. Draw dashed vertical segment dividers and alternative background shading
for idx in range(len(phase_boundaries) - 1):
    start = phase_boundaries[idx]
    end = phase_boundaries[idx + 1]
    
    # Draw vertical bounding line
    plt.axvline(start, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Apply subtle alternating gray striping background for crisp scannability
    if idx % 2 == 0:
        plt.axvspan(start, end, color='#f5f5f5', alpha=0.5, zorder=0)
    else:
        plt.axvspan(start, end, color='#ffffff', alpha=1.0, zorder=0)
        
    # Calculate the midpoints to centered-align labels along the top axis ceiling
    x_text_pos = start + (end - start) / 2
    y_text_pos = np.max(Little_curve) * 0.85
    plt.text(x_text_pos, y_text_pos, phase_labels[idx], 
             ha='center', va='center', fontsize=9, fontweight='semibold', color='#333333')

# Final vertical line at the layout boundary
plt.axvline(phase_boundaries[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

# 6. Styling adjustments
plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
plt.ylabel('ERD / ERS', fontsize=11, fontweight='bold')
plt.title('beta ERDS (Phase Structured) - Right Little', fontsize=13, fontweight='bold', pad=15)

# Tighten limits around your epoch bounds
plt.xlim([-1.0, 12.0])

# Clean layout with upper-right corner legend configuration
plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
plt.grid(False) # Turn off standard grid grids to keep phase shading clear

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# TOPOGRAPHY HEATMAPS -                                         ( L I T T L E )
#------------------------------------------------------------------------------











