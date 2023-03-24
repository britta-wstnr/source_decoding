""" Project settings for the source decoding project."""
import getpass
import numpy as np
import os.path as op
from mne.datasets import sample

# get user (server or local machine)
user = getpass.getuser()

# base paths are different on server and local machine
if user == 'we':
    code_base_path = 'path/to/python_code_base/'
    dir_out = 'path/to/output/'
elif user == 'britta':
    code_base_path = 'path/to/python_code_base/'
    dir_out = 'path/to/output/'
elif user == 'brittawe':
    code_base_path = 'path/to//python_code_base'
    dir_out = 'path/to/source_dec/output'
else:
    raise ValueError('Unknown user %s.' % user)

# further directories
dir_data_out = op.join(dir_out, 'data')
dir_sims_out = op.join(dir_out, 'simulations')

# plotting
colors_qual = [[27., 158., 119.],
               [217., 95., 2.],
               [117., 112., 179.],
               [230., 171., 2.]]
colors_qual[:] = [[x / 255. for x in row] for row in colors_qual]

colors_wes = ['#01abe9', '#1b346c', '#f54b1a']

n_trials = 200  # the number of trials per realization (to be classified)

# not for decoding_stats
snrs = np.arange(-45, 10, 5)

# For DECODING_STATS.PY only:
num_sims = 200  # how many realization per SNR in the stats decoding
n_jobs = 1  # how many cores for the stats script


# -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
# variables to change below

# VIS vs VIS  or VIS vs AUD
vis_vis = False

if vis_vis is True:
    label_names = ["Vis-lh.label", "Vis-rh.label"]
    contrast = 'vis_vis'
else:
    label_names = ["Vis-lh.label", "Aud-rh.label"]
    contrast = 'vis_aud'

# FORWARD MODEL
coreg_error = False

if coreg_error is True:
    fwd_fname = op.join(dir_data_out, 'sample_coreg_error-fwd.fif')
    coreg = 'coreg_fwd'
else:
    data_path = op.join(sample.data_path(), '/MEG/sample/')
    fwd_fname = op.join(data_path, 'sample_audvis-meg-vol-7-fwd.fif')
    coreg = 'perfect_fwd'

freqs = (72., 75.)

# filtering freqs
freq_low = 55.
freq_high = 95.
