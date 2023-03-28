# The best of two worlds: Decoding and source-reconstructing M/EEG oscillatory activity with a unique model

Britta U. Westner & Jean-Rémi King


## Repository
This repository contains the simulation and analysis code for the project _The best of two worlds: Decoding and source-reconstructing M/EEG oscillatory activity with a unique model_.

This paper is soon available as a [preprint](https://doi.org/10.1101/2023.03.24.534080).

## Code

The code in this repository is organized as follows:

### Configuration file
* `project_settings.py`

### Simulations
* `sensor_decoding.py`  - runs one realization of sensor space decoding
* `source_decoding.py`  - runs one realization of source space decoding
* `decoding_stats.py`  - runs the whole simulation with 200 realizations, takes `snr` as input, to be run in parallel for SNRs
* `decoding_stats_cv.py` - runs a grid search for optimal C parameter, 200 realizations, to be run in parallel for SNRs

### Real data analysis
* `real_data_faces.py` - analysis on the `faces` data set as shipped with MNE-Python

### Generate figures and tables
Figures 2, 4, and 5 as well as Tables 1 and 2 are generated using the Jupyter Notebooks in the subfolder `jupyter`:
* `plotting_statistics_results.ipynb`  - Fig. 2
* `plotting_statistics_results_gridsearch.ipynb` - Fig. 4 and 5

Figure 3 is generated using the following code:
* `sensor_decoding.py`
* `source_decoding.py` 

Figure 6 is generated using the real data analysis script:
* `real_data_faces.py`

## Dependencies
This work uses [MNE-Python](https://mne.tools/stable/index.html) and [scikit-learn](https://scikit-learn.org/stable).  
It further relies on a library of functions that can be found under: https://github.com/britta-wstnr/python_code_base  
For visualization, [NiBabel](https://nipy.org/nibabel/gettingstarted.html) and [Matplotlib](https://matplotlib.org/) are used.
