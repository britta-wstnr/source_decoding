import numpy as np
from mne.minimum_norm.inverse import _get_vertno
from mne.source_estimate import _make_stc, _get_src_type
from mne.forward import _subject_from_forward


def make_fake_stc(fwd, pattern):
    """Generate a fake stc for pattern to plot in source space."""

    verts = _get_vertno(fwd['src'])
    src_type = _get_src_type(fwd['src'], verts)
    subject = _subject_from_forward(fwd)
    # the transpose is only needed to get time, space dims right
    stc = _make_stc(data=np.vstack((pattern, pattern)).T,
                    vertices=verts, src_type=src_type,
                    tmin=0., tstep=1., subject=subject)

    return stc
