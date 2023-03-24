# general toolboxes
import os
import os.path as op
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# mne
import mne
from mne import read_forward_solution
from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.datasets import sample
from mne.decoding import LinearModel

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import time

# set extra paths
import sys
from project_settings import code_base_path
sys.path.insert(0, code_base_path)

# my own scripts
from csp_beamforming import beamform_components, beamform_pattern  # noqa
from generate_data import generate_data  # noqa
from matrix_transforms import stc_2_mgzvol, get_coord_from_peak  # noqa
from plot_prep import make_fake_stc  # noqa
from plotting import plot_source_act  # noqa
from project_settings import (dir_out, dir_sims_out, dir_data_out,
                              freqs, n_trials, snrs,
                              label_names, contrast, fwd_fname, coreg,
                              freq_low, freq_high) # noqa
from process_raw_data import compute_covariance  # noqa
from source_space_decoding import get_pattern  # noqa
from spatial_filtering import compute_activity_spread  # noqa
from transforms import lcmvEpochs  # noqa

# silence MNE
mne.set_log_level('WARNING')

start_all = time.time()

# ## Load everything for simulations
# read and manipulate info
data_path = sample.data_path() + '/MEG/sample/'
os.environ["SUBJECTS_DIR"] = (data_path + '../../subjects')

# read forward solution
fwd = read_forward_solution(fwd_fname, verbose=False)

# read MRI
mgz_fname = data_path + '../../subjects/sample/mri/T1.mgz'
mri_mgz = nib.load(mgz_fname)

# preallocation:
# source decoding
source_scores = []
source_scores = []
source_coords = []
source_spread = []

# beamformed patterns
source_logr_coords = []
source_logr_spread = []
source_csp_coords = []
source_csp_spread = []
source_csp_enh_coords = []
source_csp_enh_spread = []

cv = StratifiedKFold(5, random_state=0)
for snr in snrs:
    # simulate the data
    X, y, sim_coords, evokeds = generate_data(label_names=label_names,
                                              n_trials=n_trials,
                                              freqs=freqs, snr=snr,
                                              pred_filter=True,
                                              filt_def=(freq_low, freq_high),
                                              phase_lock=False)

    # #########################################################################
    # DECODE IN SOURCE SPACE - LOG REG -  NO BEAMFORMER CV

    # source reconstruct all epochs
    epochs = mne.EpochsArray(X, evokeds[0].info, tmin=0., verbose=False)

    data_cov, noise_cov = compute_covariance(epochs, t_win=(0., 0.7),
                                             t_win_noise=(0.8, 1.0),
                                             noise=True)
    spat_filter = make_lcmv(evokeds[0].info, fwd, reg=0.01, data_cov=data_cov,
                            noise_cov=noise_cov, pick_ori='max-power')

    # #########################################################################
    # DECODE IN SOURCE SPACE - LOG REG - BEAMFORMER CROSS-VALIDATED

    # beamformer is in decoding pipeline
    clf = make_pipeline(lcmvEpochs(evokeds[0].info, fwd, t_win=[0., 0.7],
                                   t_win_noise=[0.8, 1.], tmin=0., reg=0.01,
                                   erp=False),
                        StandardScaler(), LinearModel(LogisticRegression(
                            solver='liblinear')))
    scores = cross_val_score(clf, epochs._data, y, scoring='roc_auc',
                             cv=cv)

    print('Decoding scores in source space: %.2f (+/- %.2f)'
          % (np.mean(scores), np.std(scores)))
    source_scores.append((np.mean(scores), np.std(scores)))

    # #########################################################################
    # PLOT THE OBTAINED PATTERN (THAT IS THE CV'ED ONE)

    source_pattern = get_pattern(epochs._data, y, clf)
    stc = make_fake_stc(fwd, source_pattern)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)
    source_coords.append(coords_peak)

    # get the spread
    source_spread.append(compute_activity_spread(stc, fwd, threshold=0.7))

    # plot pattern onto MRI
    fig_fname = op.join(dir_sims_out,
                        'source_pattern_marker_%.1f_%s_%s.png'
                        % (snr, contrast, coreg))
    plot_source_act(stc, fwd, mri=mgz_fname, timepoint=0, threshold=0.7,
                    thresh_ref='timepoint', cmap='coolwarm',
                    save_fig=False, fig_fname=fig_fname, display_mode='z',
                    add_coords=True, coords=sim_coords)
    plt.savefig(fig_fname)

    # #########################################################################
    # CREATE A SOURCE PATTERN FROM THE SENSOR LOGREG DECODING

    sensor_pattern_logr = np.load(op.join(dir_data_out,
                                          'sensor_pattern_logr_%.3f_%s.npy'
                                          % (snr, contrast)))

    # beamform the pattern using spatial filter from above
    stc_logr = beamform_pattern(sensor_pattern_logr, spat_filter, fwd)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc_logr, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)
    source_logr_coords.append(coords_peak)

    # get the spread
    source_logr_spread.append(compute_activity_spread(stc_logr, fwd,
                                                      threshold=0.7))

    # plot beamformed sensor pattern
    fig_fname = op.join(dir_sims_out, 'beamf_pattern_%.1f_%s_%s.png'
                        % (snr, contrast, coreg))
    plot_source_act(stc_logr, fwd, mri=mgz_fname, timepoint=0, threshold=0.7,
                    thresh_ref='timepoint', cmap='coolwarm',
                    save_fig=False, fig_fname=fig_fname, display_mode='z',
                    add_coords=True, coords=sim_coords)
    plt.savefig(fig_fname)

    # #########################################################################
    # CREATE SOURCE PATTERN FROM CSP COMPONENTS

    weights_csp = np.load(op.join(dir_data_out,
                                  'sensor_weights_csp_%.3f_%s.npy'
                                  % (snr, contrast)))
    pattern_csp = np.load(op.join(dir_data_out,
                                  'sensor_pattern_csp_%.3f_%s.npy'
                                  % (snr, contrast)))

    # Use both multipliers and not multipliers
    multiplier = (1, -1, 1, 1)

    # Combine the components
    stc_csp = beamform_components(weights_csp, pattern_csp, spat_filter,
                                  fwd, multipliers=multiplier)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc_csp, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)

    # peak coordinates and spread:
    source_csp_coords.append(coords_peak)
    # get the spread
    source_csp_spread.append(compute_activity_spread(stc_csp, fwd,
                                                     threshold=0.7))

    # file name
    multi = '_multiplied' if multiplier is not None else ''
    fig_fname = op.join(dir_sims_out,
                        'beamf_csp_pattern_%.1f_%s_%s_%s.png'
                        % (snr, contrast, coreg, multi))
    plot_source_act(stc_csp, fwd, mri=mgz_fname, timepoint=0,
                    threshold=0.5,
                    thresh_ref='timepoint', cmap='coolwarm',
                    save_fig=False, fig_fname=fig_fname, display_mode='z',
                    add_coords=True, coords=sim_coords)
    plt.savefig(fig_fname)

    plt.close('all')

    # #########################################################################
    # PLOT THE ACTUAL SOURCE POWER
    # for visualization purposes in the paper
    stcs = apply_lcmv_epochs(epochs, spat_filter, return_generator=True,
                             max_ori_out='signed')
    time_idx_a = epochs.time_as_index(0.)
    time_idx_b = epochs.time_as_index(0.7)
    stcs_mat = np.ones((X.shape[0], fwd['nsource']))
    for trial in range(X.shape[0]):
        stcs_mat[trial, :] = np.mean(
            next(stcs).data[:, time_idx_a[0]:time_idx_b[0]] ** 2, axis=1)

    source_one = np.mean(stcs_mat[0:200, :], axis=0)
    stc_one = make_fake_stc(fwd, source_one)
    source_two = np.mean(stcs_mat[200:, :], axis=0)
    stc_two = make_fake_stc(fwd, source_two)

    fig_fname = op.join(dir_sims_out,
                        'source_pattern_one_%.1f_%s.png'
                        % (snr, contrast))
    plot_source_act(stc_one, fwd, mri=mgz_fname, timepoint=0,
                    threshold=0.6,
                    thresh_ref='timepoint', cmap='magma',
                    save_fig=False, fig_fname=fig_fname, display_mode='z',
                    add_coords=False, coords=None)
    plt.savefig(fig_fname)

    fig_fname = op.join(dir_sims_out,
                        'source_pattern_two_%.1f_%s.png'
                        % (snr, contrast))
    plot_source_act(stc_two, fwd, mri=mgz_fname, timepoint=0,
                    threshold=0.6,
                    thresh_ref='timepoint', cmap='magma',
                    save_fig=False, fig_fname=fig_fname, display_mode='z',
                    add_coords=False, coords=None)
    plt.savefig(fig_fname)
    plt.close('all')

# save scores
np.save(op.join(dir_data_out, 'source_acc_%s_%s' % (contrast, coreg)),
        source_scores)

# plot localization error
fig_fname = op.join(dir_out, 'local_errors_%s_%s.eps' % (contrast, coreg))
# loop over source decoding vs CSP
for coords in ([source_coords, source_csp_coords]):
    error_left = [np.linalg.norm((sim_coords[0] - x))
                  for x in coords]
    error_right = [np.linalg.norm((sim_coords[1] - x))
                   for x in coords]
    plt.plot(snrs, np.min((error_left, error_right), axis=0),
             marker='s', linewidth=2, markersize=10)
plt.xlim(-0.1, 1.1)
plt.xlabel('Input SNR')
plt.ylabel('Localization error (mm)')
plt.title('Localization error as a function of SNR')
plt.legend(('Source', 'CSP'))
plt.savefig(fig_fname)

plt.close('all')

# plot activation spreads
fig_fname = op.join(dir_out, 'act_spread_%s_%s.eps' % (contrast, coreg))
for spr in [source_spread, source_csp_spread]:
    plt.plot(snrs, spr, marker='s', linewidth=2, markersize=10)
plt.xlim(-0.1, 1.1)
plt.xlabel('Input SNR')
plt.ylabel('Source activity spread (a. u.)')
plt.title('Source spread as a function of SNR')
plt.legend(('Source', 'CSP'))
plt.savefig(fig_fname)

plt.close("all")

end_all = time.time()
print('Total elapsed time: ' + time.strftime("%H:%M:%S",
                                             time.gmtime(end_all - start_all)))
