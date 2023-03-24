# general toolboxes
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

# mne
import mne
from mne.cov import compute_covariance
from mne.datasets import sample
from mne.decoding import CSP, LinearModel

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# set extra paths
import sys
from project_settings import code_base_path
sys.path.insert(0, code_base_path)

# scripts and vars from this project
from project_settings import (dir_data_out, dir_sims_out,
                              freqs, n_trials, snrs, freq_low, freq_high,
                              label_names, contrast) # noqa
from generate_data import generate_data, get_power_or_erp  # noqa

# scripts from Python Code Base
from simulations import generate_signal, simulate_evoked_osc  # noqa
from process_raw_data import compute_covariance  # noqa
from spatial_filtering import run_lcmv_epochs  # noqa
from transforms import lcmvEpochs  # noqa
from source_space_decoding import get_pattern  # noqa
from plotting import plot_source_act  # noqa
from matrix_transforms import stc_2_mgzvol  # noqa
from signal_processing import get_max_diff, estimate_snr  # noqa

# silence MNE
mne.set_log_level('WARNING')

# #############################################################################
# LOAD INGREDIENTS
data_path = sample.data_path() + '/MEG/sample/'
os.environ["SUBJECTS_DIR"] = (data_path + '../../subjects')

# preallocation
sensor_scores = []
csp_scores = []
effective_snr = []

for snr in snrs:
    # simulate the data
    X, y, sim_coords, evokeds = generate_data(label_names=label_names,
                                              n_trials=n_trials,
                                              freqs=freqs, snr=snr,
                                              pred_filter=True,
                                              filt_def=(freq_low, freq_high),
                                              phase_lock=False)

    epochs = mne.EpochsArray(X, evokeds[0].info, tmin=0., verbose=False)

    activity = get_power_or_erp(epochs._data, evokeds, phase_lock=False,
                                power_win=(0., 0.7))
    effective_snr.append(estimate_snr(epochs, (100, 150), (0, 50)))

    # plot the power in the two conditions
    power_one = np.mean(activity[:200, ], axis=0)
    power_two = np.mean(activity[200:, ], axis=0)

    clim = np.max(np.abs((np.min((power_one, power_two)),
                         np.max((power_one, power_two)))))

    fig_fname = op.join(dir_sims_out, 'evoked_one_%.3f_%s.png'
                        % (snr, contrast))
    fig, ax = plt.subplots()
    tp = mne.viz.plot_topomap(power_one, evokeds[0].info,
                              sensors=False, vmin=0, vmax=clim,
                              cmap=plt.cm.plasma)
    plt.plasma()
    cbar = ax.figure.colorbar(tp[0], ax=ax)
    cbar.ax.set_ylabel('Pattern (a.u.)', rotation=90, va='bottom')
    plt.savefig(fig_fname)

    fig_fname = op.join(dir_sims_out, 'evoked_two_%.3f_%s.png'
                        % (snr, contrast))
    fig, ax = plt.subplots()
    tp = mne.viz.plot_topomap(power_two, evokeds[0].info,
                              sensors=False, vmin=0, vmax=clim,
                              cmap=plt.cm.plasma)
    cbar = ax.figure.colorbar(tp[0], ax=ax)
    cbar.ax.set_ylabel('Pattern (a.u.)', rotation=90, va='bottom')
    plt.savefig(fig_fname)
    plt.close('all')

    # #########################################################################
    # DECODE POWER IN SENSOR SPACE - LOG REG
    cv = StratifiedKFold(5, random_state=0)

    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(
        solver='liblinear')))
    scores = cross_val_score(clf, activity, y, scoring='roc_auc', cv=cv)

    print('Decoding scores LogReg: %.2f (+/- %.2f)'
          % (np.mean(scores), np.std(scores)))
    sensor_scores.append((np.mean(scores), np.std(scores)))

    # get the patterns and save
    sensor_pattern_logr = get_pattern(activity, y, clf)
    np.save(op.join(dir_data_out, 'sensor_pattern_logr_%.3f_%s'
                    % (snr, contrast)),
            sensor_pattern_logr)

    # plot pattern
    fig, ax = plt.subplots()
    fig_fname = op.join(dir_sims_out, 'sensor_pattern_%.3f_%s.png'
                        % (snr, contrast))
    clim = np.max(np.abs((sensor_pattern_logr.min(),
                          sensor_pattern_logr.max())))
    tp = mne.viz.plot_topomap(sensor_pattern_logr, evokeds[0].info,
                              sensors=False, vmin=-clim, vmax=clim)
    cbar = ax.figure.colorbar(tp[0], ax=ax)
    cbar.ax.set_ylabel('Pattern (a.u.)', rotation=90, va='bottom')
    plt.savefig(fig_fname)
    plt.close('all')

    # #########################################################################
    # DECODE CSP AND LOG REG
    # make the CSP model

    epochs.crop(0., 0.7)

    n_comp = 4
    csp = CSP(norm_trace=False, transform_into='average_power', log=False,
              n_components=n_comp)

    # make the pipeline and classify the data
    csp_clf = make_pipeline(csp, LinearModel(LogisticRegression(
        solver='liblinear')))
    scores = cross_val_score(csp_clf, epochs._data, y, scoring='roc_auc',
                             cv=cv)
    weights_csp = get_pattern(epochs._data, y, csp_clf)

    print('Decoding scores CSP: %.3f (+/- %.3f)' %
          (np.mean(scores), np.std(scores)))
    csp_scores.append((np.mean(scores), np.std(scores)))

    # Plot the pattern from CSP +  logistic regression
    # re-fit the model on all data and get the pattern
    csp.fit_transform(epochs._data, y)
    sensor_pattern_csp = csp.patterns_

    # plot the patterns
    fig_fname = op.join(dir_sims_out, 'csp_pattern_%.3f_%s.png'
                        % (snr, contrast))
    clim = np.max(np.abs((csp.patterns_[0:n_comp].min(),
                          csp.patterns_[0:n_comp].max())))
    fig = csp.plot_patterns(evokeds[0].info, colorbar=True);  # noqa
    plt.savefig(fig_fname)
    plt.close('all')

    # save CSP
    np.save(op.join(dir_data_out, 'sensor_pattern_csp_%.3f_%s'
                    % (snr, contrast)),
            csp.patterns_)
    np.save(op.join(dir_data_out, 'sensor_weights_csp_%.3f_%s'
                    % (snr, contrast)),
            weights_csp)


# save scores
np.save(op.join(dir_data_out, 'sensor_logr_acc_%s' % contrast),
        sensor_scores)
np.save(op.join(dir_data_out, 'sensor_csp_acc_%s' % contrast),
        csp_scores)
