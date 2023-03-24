# general toolboxes
import os
import os.path as op
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# mne
import mne
from mne.beamformer import make_lcmv
from mne.datasets import spm_face
from mne.decoding import LinearModel, CSP

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# set extra paths
import sys
from project_settings import code_base_path
if 'python_code_base' not in sys.path[0]:
    sys.path.insert(0, code_base_path)

# my own scripts
from csp_beamforming import beamform_components  # noqa
from plot_prep import make_fake_stc  # noqa
from plotting import plot_source_act  # noqa
from process_raw_data import compute_covariance  # noqa
from project_settings import (dir_data_out, dir_sims_out, colors_wes)  # noqa
from source_space_decoding import get_pattern  # noqa
from transforms import lcmvEpochs  # noqa

# silence MNE
mne.set_log_level('WARNING')


# ## Load everything for analysis
data_path = spm_face.data_path()
meg_data_path = op.join(data_path, 'MEG/spm/')
os.environ["SUBJECTS_DIR"] = op.join(data_path, '/subjects')

# read data and epoch
raw_fname = op.join(meg_data_path, 'SPM_CTF_MEG_example_faces1_3D.ds')
raw = mne.io.read_raw_ctf(raw_fname, preload=True)


# read MRI
mgz_fname = op.join(data_path, 'subjects/spm/mri/T1.mgz')
mri_mgz = nib.load(mgz_fname)

# read or compute forward model
fwd_fname = op.join(dir_data_out, 'spm_faces_5mm_fwd.fif')

if not op.isfile(fwd_fname):

    # read BEM model
    bem = op.join(data_path, 'subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif')

    # volume source grid
    src = mne.setup_volume_source_space('spm', pos=5., mri=mgz_fname, bem=bem)

    # make forward model
    trans_fname = meg_data_path + 'SPM_CTF_MEG_example_faces1_3D_raw-trans.fif'
    fwd = mne.make_forward_solution(raw.info, trans=trans_fname, src=src,
                                    bem=bem, meg=True, eeg=False, n_jobs=2)

    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
else:
    fwd = mne.read_forward_solution(fwd_fname)

# set up events
events = mne.find_events(raw, stim_channel='UPPT001')
event_ids = {"faces": 1, "scrambled": 2}

# pick channels
raw.del_proj()
picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                       exclude='bads', ref_meg=False)
raw.pick_channels([raw.ch_names[pick] for pick in picks])


contrast = 'faces'

# Settings for time windows, epochs are cut -0.5 to 0.5
noise_min, noise_max = -0.45, -0.15
act_min, act_max = 0.15, 0.45
power_min, power_max = 0.18, 0.4

# beamformer settings
pick_ori = 'max-power'
reg = 0.01
weight_norm = 'unit-noise-gain'

# frequency settings
freq_min, freq_max = 60., 95.

event_id, tmin, tmax = [1, 2], -0.6, 0.6

# run actual event selection and epoching:
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    baseline=(None, 0), preload=True, proj=True)

erp_save = 'power'

# #########################################################################
# DECODE IN SENSOR SPACE - LOG REG

epochs.filter(freq_min, freq_max, fir_design='firwin')
data_tmp = epochs.get_data()
# take power of 0 to 250 ms
power_win = (power_min, power_max)
time_idx = epochs.time_as_index(power_win)
X = data_tmp[:, :, time_idx[0]:time_idx[1]]
X_sens = np.mean(X ** 2, axis=2)

y = epochs.events[:, 2]

# set up pipeline for classification
cv = StratifiedKFold(5)
clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(
    solver='liblinear')))

sensor_scores = cross_val_score(clf, X_sens, y, scoring='roc_auc', cv=cv)
sensor_pattern_logr = get_pattern(X_sens, y, clf)

clim = np.max(np.abs((sensor_pattern_logr.min(),
                      sensor_pattern_logr.max())))

fig = mne.viz.plot_topomap(sensor_pattern_logr, epochs.info,
                           sensors=False, vmin=-clim, vmax=clim);  # noqa
fig_fname = op.join(dir_sims_out,
                    'realdata_sensor_patterns_%s_%s.png' % (contrast,
                                                            erp_save))
plt.savefig(fig_fname)

# #########################################################################
# DECODE IN SENSOR SPACE - CSP AND LOG-REG

n_comp = 2
csp = CSP(norm_trace=False, transform_into='average_power',
          n_components=n_comp)

# make the pipeline and classify the data
csp_clf = make_pipeline(csp, LinearModel(LogisticRegression(
    solver='liblinear')))

csp_scores = cross_val_score(csp_clf, X, y, scoring='roc_auc', cv=cv)
weights_csp = get_pattern(X, y, csp_clf)

# Plot the pattern from CSP +  logistic regression
# re-fit the model on all data and get the pattern
csp.fit_transform(X, y)
sensor_pattern_csp = csp.patterns_

# plot the patterns
clim = np.max(np.abs((csp.patterns_[0:n_comp].min(),
                      csp.patterns_[0:n_comp].max())))
fig = csp.plot_patterns(epochs.info);  # noqa
fig_fname = op.join(dir_sims_out,
                    'realdata_csp_patterns_%s_%s.png' % (contrast,
                                                         erp_save))
plt.savefig(fig_fname)
plt.close('all')

# #########################################################################
# SOURCE SPACE DECODING
clf = make_pipeline(lcmvEpochs(epochs.info, fwd, t_win=(act_min, act_max),
                               t_win_noise=(noise_min, noise_max),
                               tmin=tmin, reg=reg, pick_ori=pick_ori,
                               weight_norm=weight_norm,
                               erp=False, time_idx=time_idx,
                               power_win=(power_min, power_max)),
                    StandardScaler(), LinearModel(
                            LogisticRegression(solver='liblinear')))

# act_source = np.mean(epochs_source[:, :, time_idx] ** 2, axis=2)
print('Source space decoding may take a while.')
source_scores = cross_val_score(clf, epochs.get_data(),
                                y, scoring='roc_auc', cv=cv)

source_pattern = get_pattern(epochs.get_data(), y, clf)
stc = make_fake_stc(fwd, source_pattern)

# plot pattern onto MRI
fig_fname = op.join(dir_sims_out,
                    'realdata_source_pattern_%s_%s.png' % (contrast,
                                                           erp_save))
plot_source_act(stc, fwd, mri=mgz_fname, timepoint=0, threshold=0.75,
                thresh_ref='timepoint', cmap=None,
                save_fig=False, display_mode='ortho')
plt.savefig(fig_fname)

# #########################################################################
# CREATE A SOURCE PATTERN FROM THE SENSOR LOGREG DECODING

# beamform the pattern using a similar spatial filter to above
data_cov, noise_cov = compute_covariance(epochs, t_win=(act_min, act_max),
                                         t_win_noise=(noise_min,
                                                      noise_max),
                                         noise=True)
spat_filter = make_lcmv(epochs.info, fwd, reg=reg, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori=pick_ori,
                        weight_norm=weight_norm)

whitened_pattern = np.dot(spat_filter['whitener'], sensor_pattern_logr.T)
beamformed_pattern = np.dot(spat_filter['weights'], whitened_pattern)
stc_logr = make_fake_stc(fwd, beamformed_pattern)

# plot beamformed sensor pattern
stc_logr.data[:, :] = np.vstack(beamformed_pattern)

fig_fname = op.join(dir_sims_out,
                    'realdata_beamf_pattern_%s_%s.png' % (contrast,
                                                          erp_save))
plot_source_act(stc_logr, fwd, mri=mgz_fname, timepoint=0, threshold=0.75,
                thresh_ref='timepoint', cmap=None,
                save_fig=False, display_mode='ortho')
plt.savefig(fig_fname)
plt.close('all')

# #########################################################################
# CREATE SOURCE PATTERN FROM CSP COMPONENTS

# Combine the components
multipliers = (1, -1)
stc_csp = beamform_components(weights_csp, sensor_pattern_csp, spat_filter,
                              fwd, multipliers=multipliers)

# plotting:
fig_fname = op.join(dir_sims_out,
                    'realdata_beamf_csp_pattern_%s_%s.png'
                    % (contrast, erp_save))
plot_source_act(stc_csp, fwd, mri=mgz_fname, timepoint=0,
                threshold=0.85,
                thresh_ref='timepoint', cmap=None,
                save_fig=False, display_mode='ortho')
plt.savefig(fig_fname)

plt.close('all')

# #############################################################################
# PRINT THE DECODING SCORES TO TERMINAL

print('Decoding scores: \n Sensor decoding: %f \n Source decoding: %f \n'
      ' CSP sensor decoding: %f' % (np.mean(sensor_scores),
                                    np.mean(source_scores),
                                    np.mean(csp_scores)))

# #############################################################################
# PLOT THE DECODING ACCURACIES

scores_data = [np.mean(sensor_scores), np.mean(source_scores),
               np.mean(csp_scores)]
scores_std = [np.std(sensor_scores), np.std(source_scores),
               np.std(csp_scores)]
positions = range(3)

plt.figure()
fig, ax = plt.subplots()

for score, std, pos, col in zip(scores_data, scores_std, positions, colors_wes):
    plot_score = score - 0.5#

    if plot_score < 0.5:
        lowlim = True
    else:
        lowlim = False
    uplim = not lowlim 

    ax.bar(pos, height=plot_score, color=col, yerr=std, ecolor=col, 
    capsize=15, linewidth=2, error_kw={uplims=uplim, lowlims=lowlim})
    # ax.errorbar(pos, plot_score, yerr=std, ecolor='black', elinewidth=2)

ax.set_xticks(positions)
ax.set_xticklabels(['Sensor space', 'Source space', 'CSP'])
ax.tick_params(axis='x', length=0)  # prevent x ticks from showing

ax.set_ylim(-0.075, 0.175)
ax.set_yticks(np.arange(-0.05, 0.2, 0.05))
ax.set_yticklabels(np.arange(45, 70, 5))

plt.axhline(y=0, color='black')
plt.ylabel('Decoding score (ROC AUC)')
plt.title('Real data performance')

# note: chance levels
# 0.001: 61.9048 | 0.01: 58.9286 | 0.05: 56.5476

fig_fname = op.join(dir_sims_out, 'realdata_decoding_scores_%s.pdf' % erp_save)
plt.savefig(fig_fname)
