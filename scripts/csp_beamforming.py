import numpy as np

# own scripts
import sys
from project_settings import code_base_path
if 'python_code_base' not in sys.path[0]:
    sys.path.insert(0, code_base_path)
from plot_prep import make_fake_stc  # noqa


def beamform_components(components, sensor_pattern, spat_filter, fwd,
                        multipliers=None):
    """Beamform CSP components and regression weights after sensor decoding."""

    if multipliers is None:
        for comp in range(len(components)):
            if comp == 0:
                sensor_pattern_csp = sensor_pattern[comp] * components[comp]
            else:
                sensor_pattern_csp += sensor_pattern[comp] * components[comp]

        # beamform the pattern
        whitened_pattern = np.dot(spat_filter['whitener'],
                                  sensor_pattern_csp.T)
        beamformed_pattern = np.dot(spat_filter['weights'], whitened_pattern)

    else:
        # Check multiplier object
        if len(multipliers) != len(components):
            raise ValueError('Number of supplied multipliers needs to match '
                             'number of CSP components.')
        elif not isinstance(multipliers, tuple):
            raise ValueError('multipliers should be a tuple.')

        # Beamform pattern one after the other and multiply with multiplier in
        # source space
        for ii, mult in enumerate(multipliers):
            sensor_pattern_csp = sensor_pattern[ii] * components[ii]
            whitened_pattern = np.dot(spat_filter['whitener'],
                                      sensor_pattern_csp.T)
            interim_pattern = np.dot(spat_filter['weights'],
                                     whitened_pattern)
            if ii == 0:
                beamformed_pattern = mult * abs(interim_pattern)
            else:
                beamformed_pattern += mult * abs(interim_pattern)

    # make fake stc
    beamformed_pattern = make_fake_stc(fwd, beamformed_pattern)

    return beamformed_pattern


def beamform_pattern(sensor_pattern, spat_filter, fwd):
    """Beamform sensor pattern after sensor decoding.

    DISCLAIMER: this will probably fail for most data, consider using
    the function `beamform_components()` instead.
    """
    whitened_pattern = np.dot(spat_filter['whitener'], sensor_pattern.T)
    beamformed_pattern = np.dot(spat_filter['weights'], whitened_pattern)

    # make face stc
    beamformed_pattern = make_fake_stc(fwd, beamformed_pattern)

    return beamformed_pattern
