# general toolboxes
import os
import os.path as op
import nibabel as nib
import numpy as np
import time

# mne
import mne
from mne import read_forward_solution
from mne.beamformer import make_lcmv
from mne.cov import compute_covariance
from mne.datasets import sample
from mne.decoding import CSP, LinearModel

# scikit-learn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# set extra paths
import sys
from project_settings import code_base_path
sys.path.insert(0, code_base_path)

# scripts and vars from this project
from project_settings import (dir_data_out, fwd_fname, n_jobs,
                              freqs, n_trials, freq_low, freq_high,
                              label_names, contrast,
                              num_sims, vis_vis) # noqa
from plot_prep import make_fake_stc  # noqa

# scripts from Python Code Base
from csp_beamforming import beamform_components, beamform_pattern  # noqa
from generate_data import generate_data, get_power_or_erp  # noqa
from simulations import generate_signal, simulate_evoked_osc  # noqa
from process_raw_data import compute_covariance  # noqa
from spatial_filtering import run_lcmv_epochs, compute_activity_spread  # noqa
from transforms import lcmvEpochs  # noqa
from source_space_decoding import get_pattern  # noqa
from plotting import plot_source_act  # noqa
from matrix_transforms import stc_2_mgzvol, get_coord_from_peak  # noqa
from signal_processing import get_max_diff, estimate_snr  # noqa

# silence MNE
mne.set_log_level('WARNING')

# get command line input
snr = float(sys.argv[1])

# get start time
start_all = time.time()

# #############################################################################
# LOAD INGREDIENTS
data_path = sample.data_path() + '/MEG/sample/'
os.environ["SUBJECTS_DIR"] = (data_path + '../../subjects')

# read forward solution
fwd = read_forward_solution(fwd_fname, verbose=False)

# read MRI
mgz_fname = data_path + '../../subjects/sample/mri/T1.mgz'
mri_mgz = nib.load(mgz_fname)

sim_coords = []
effective_snr = []
# prealloc sensors
sensor_scores = []
sensor_time = []
sensor_coords = []
sensor_spread = []
# prealloc csp
csp_scores = []
csp_spread = []
csp_coords = []
csp_time = []
# prealloc source
source_scores = []
source_spread = []
source_coords = []
source_time = []

if vis_vis is False:
    # only for auditory-visual: do grid search of C
    sensor_grid_c = []
    source_grid_c = []
    csp_grid_c = []
    source_scores_c = []
    sensor_scores_c = []
    csp_scores_c = []

for my_seed in range(num_sims):
    # simulate the data
    X, y, sim_c_ii, evokeds = generate_data(label_names=label_names,
                                            n_trials=n_trials, freqs=freqs,
                                            snr=snr, pred_filter=True,
                                            filt_def=(freq_low, freq_high),
                                            phase_lock=False, loc='random')
    sim_coords.append(sim_c_ii)

    epochs = mne.EpochsArray(X, evokeds[0].info, verbose=False)

    cv = StratifiedKFold(5, random_state=0)

    activity = get_power_or_erp(epochs._data, evokeds, phase_lock=False,
                                power_win=(0., 0.7))
    effective_snr.append(estimate_snr(epochs, (100, 150), (0, 50)))

    # #########################################################################
    # SOURCE space Logistic Regression
    data_cov, noise_cov = compute_covariance(epochs, t_win=(0., 0.7),
                                             t_win_noise=(0.8, 1.0),
                                             noise=True)
    spat_filter = make_lcmv(evokeds[0].info, fwd, reg=0.01,
                            data_cov=data_cov,
                            noise_cov=noise_cov, pick_ori='max-power')

    source_start = time.time()
    clf = make_pipeline(lcmvEpochs(evokeds[0].info, fwd, t_win=[0., 0.7],
                                   t_win_noise=[0.8, 1.], tmin=0.,
                                   reg=0.01, erp=False),
                        StandardScaler(), LinearModel(LogisticRegression(
                            solver='liblinear')))
    scores = cross_val_score(clf, epochs._data, y, scoring='roc_auc',
                             cv=cv)
    source_stop = time.time()

    source_scores.append(np.mean(scores))
    source_time.append(source_stop - source_start)

    # SOURCE space coordinates and spread
    source_pattern = get_pattern(epochs._data, y, clf)
    stc = make_fake_stc(fwd, source_pattern)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)
    source_coords.append(coords_peak)

    # get the spread
    source_spread.append(compute_activity_spread(stc, fwd, threshold=0.7))

    if vis_vis is False:  # also do grid search for C
        # STEP 1: grid search C
        clf = make_pipeline(lcmvEpochs(evokeds[0].info, fwd, t_win=[0., 0.7],
                                       t_win_noise=[0.8, 1.], tmin=0.,
                                       reg=0.01, erp=False),
                            StandardScaler(), LinearModel(LogisticRegressionCV(
                                solver='liblinear', cv=cv)))

        clf.fit(epochs._data, y)
        best_c = clf.named_steps['linearmodel'].model.C_
        source_grid_c.append(best_c)

        # STEP 2: fit with best C
        clf = make_pipeline(lcmvEpochs(evokeds[0].info, fwd, t_win=[0., 0.7],
                                       t_win_noise=[0.8, 1.], tmin=0.,
                                       reg=0.01, erp=False),
                            StandardScaler(), LinearModel(LogisticRegression(
                                solver='liblinear', C=best_c[0])))
        scores = cross_val_score(clf, epochs._data, y, scoring='roc_auc',
                                 cv=cv)
        source_scores_c.append((np.mean(scores)))

    # #########################################################################
    # SENSOR space Logistic Regression
    sensor_start = time.time()

    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(
        solver='liblinear')))
    scores = cross_val_score(clf, activity, y, scoring='roc_auc', cv=cv,
                             n_jobs=n_jobs)
    sensor_stop = time.time()

    # sensor scores
    sensor_scores.append(np.mean(scores))

    sensor_time.append(sensor_stop - sensor_start)

    # create a source space pattern from the sensor patterns
    sensor_pattern = get_pattern(activity, y, clf)
    stc_logr = beamform_pattern(sensor_pattern, spat_filter, fwd)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc_logr, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)
    sensor_coords.append(coords_peak)

    # get the spread
    sensor_spread.append(compute_activity_spread(stc_logr, fwd, threshold=0.7))

    if vis_vis is False:  # also do grid search for C
        # STEP 1: grid search C
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegressionCV(
            solver='liblinear', cv=cv)))
        clf.fit(activity, y)
        best_c = clf.named_steps['linearmodel'].model.C_
        sensor_grid_c.append(best_c)

        # STEP 2: fit with best C
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(
            solver='liblinear', C=best_c[0])))
        scores = cross_val_score(clf, activity, y, scoring='roc_auc', cv=cv,
                                 n_jobs=n_jobs)

        sensor_scores_c.append(np.mean(scores))

    # #########################################################################
    # CSP SENSOR space
    # sensor space CSP + Logistic Regression
    n_comp = 4
    csp_start = time.time()
    csp = CSP(norm_trace=False, transform_into='average_power',
              n_components=n_comp)

    # crop the data
    epochs.crop(0., 0.7)

    # make the pipeline and classify the data
    csp_clf = make_pipeline(csp, LinearModel(LogisticRegression(
        solver='liblinear')))
    scores = cross_val_score(csp_clf, epochs._data, y,
                             scoring='roc_auc', cv=cv, n_jobs=n_jobs)
    csp_stop = time.time()

    csp_scores.append(np.mean(scores))
    csp_time.append(csp_stop - csp_start)

    # create a source space pattern from the CSP patterns
    weights_csp = get_pattern(epochs._data, y, csp_clf)
    pattern_csp = csp.patterns_

    # combine the components
    multipliers = (1, -1, 1, 1)  # enhanced CSP version
    stc_csp = beamform_components(weights_csp, pattern_csp, spat_filter,
                                  fwd, multipliers=multipliers)

    # get the peak coordinates
    coords_meg = get_coord_from_peak(stc_csp, fwd)
    coords_peak = stc_2_mgzvol(coords_meg, fwd, mri_mgz)
    csp_coords.append(coords_peak)

    # get the spread
    csp_spread.append(compute_activity_spread(stc_csp, fwd, threshold=0.7))

    if vis_vis is False:  # also do grid search for C
        # STEP 1: grid search C
        csp_clf = make_pipeline(csp, LinearModel(LogisticRegressionCV(
            solver='liblinear', cv=cv)))
        csp_clf.fit(epochs._data, y)
        best_c = csp_clf.named_steps['linearmodel'].model.C_
        csp_grid_c.append(best_c)

        # STEP 2: fit with best C
        csp_clf = make_pipeline(csp, LinearModel(LogisticRegression(
            solver='liblinear', C=best_c[0])))
        scores = cross_val_score(csp_clf, epochs._data, y,
                                 scoring='roc_auc', cv=cv, n_jobs=n_jobs)
        csp_scores_c.append(np.mean(scores))


# #############################################################################
# SAVE OUTPUTS FOR THIS SNR

# save source output
np.save(op.join(dir_data_out, 'stats', 'source_scores_%s_%.3f.npy') %
        (contrast, snr), source_scores)
np.save(op.join(dir_data_out, 'stats', 'source_coords_%s_%.3f.npy') %
        (contrast, snr), source_coords)
np.save(op.join(dir_data_out, 'stats', 'source_spread_%s_%.3f.npy') %
        (contrast, snr), source_spread)
np.save(op.join(dir_data_out, 'stats', 'source_time_%s_%.3f.npy') %
        (contrast, snr), source_time)

# save sensor output
np.save(op.join(dir_data_out, 'stats', 'sensor_scores_%s_%.3f.npy') %
        (contrast, snr), sensor_scores)
np.save(op.join(dir_data_out, 'stats', 'sensor_time_%s_%.3f.npy') %
        (contrast, snr), sensor_time)
np.save(op.join(dir_data_out, 'stats', 'sensor_coords_%s_%.3f.npy') %
        (contrast, snr), sensor_coords)
np.save(op.join(dir_data_out, 'stats', 'sensor_spread_%s_%.3f.npy') %
        (contrast, snr), sensor_coords)

# save CSP output
np.save(op.join(dir_data_out, 'stats', 'csp_scores_%s_%.3f.npy') %
        (contrast, snr), csp_scores)
np.save(op.join(dir_data_out, 'stats', 'csp_coords_%s_%.3f.npy') %
        (contrast, snr), csp_coords)
np.save(op.join(dir_data_out, 'stats', 'csp_spread_%s_%.3f.npy') %
        (contrast, snr), csp_spread)
np.save(op.join(dir_data_out, 'stats', 'csp_time_%s_%.3f.npy') %
        (contrast, snr), csp_time)

# simulation coords and effective snr
np.save(op.join(dir_data_out, 'stats', 'sim_coords_%s_%.3f.npy') %
        (contrast, snr), sim_coords)
np.save(op.join(dir_data_out, 'stats', 'effective_snr_%s_%.3f.npy') %
        (contrast, snr), effective_snr)

# save grid search results for auditory-visual
if vis_vis is False:
    # save source output
    np.save(op.join(dir_data_out, 'stats', 'cv_source_scores_%s_%.3f.npy') %
            (contrast, snr), source_scores_c)
    np.save(op.join(dir_data_out, 'stats', 'cv_source_c_%s_%.3f.npy') %
            (contrast, snr), source_grid_c)
    # save sensor output
    np.save(op.join(dir_data_out, 'stats', 'cv_sensor_scores_%s_%.3f.npy') %
            (contrast, snr), sensor_scores_c)
    np.save(op.join(dir_data_out, 'stats', 'cv_sensor_c_%s_%.3f.npy') %
            (contrast, snr), sensor_grid_c)
    # save CSP output
    np.save(op.join(dir_data_out, 'stats', 'cv_csp_scores_%s_%.3f.npy') %
            (contrast, snr), csp_scores_c)
    np.save(op.join(dir_data_out, 'stats', 'cv_csp_c_%s_%.3f.npy') %
            (contrast, snr), csp_grid_c)

# Report durations:
end_all = time.time()

print('Total elapsed time: ' + time.strftime("%H:%M:%S",
                                             time.gmtime(end_all - start_all)))
