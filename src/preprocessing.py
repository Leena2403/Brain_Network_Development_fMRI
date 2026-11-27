"""
Preprocessing: load the dataset, check head motion, exclude noisy scans,
and turn each brain scan into one signal per brain region.

Used by notebooks 01 and 02.
"""

import os
import re
import time

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker

TR = 2.0  # seconds between brain volumes (Richardson et al., 2018)

# Exclusion rules, decided BEFORE looking at any results
FD_THRESHOLD = 0.5    # a person is excluded if their mean FD is above 0.5 mm ...
FRAC_THRESHOLD = 0.2  # ... or if more than 20% of their volumes have FD above 0.5 mm


def load_dataset(n_subjects=None, max_tries=10):
    """Download (first time only) the development fMRI dataset.

    Returns the nilearn data object and a table with one row per participant,
    in the same order as the scan files.

    The OSF server sometimes times out mid-download. Files that finished are
    kept on disk, so we simply try again and it continues where it stopped.
    """
    for attempt in range(1, max_tries + 1):
        try:
            data = datasets.fetch_development_fmri(n_subjects=n_subjects, verbose=0)
            break
        except Exception as err:  # network errors (timeouts, dropped connections)
            if attempt == max_tries:
                raise
            print(f"Download interrupted ({type(err).__name__}), retrying {attempt}/{max_tries - 1}...")
            time.sleep(5)
    pheno = pd.DataFrame(data.phenotypic).reset_index(drop=True)

    # Safety check: row i of the table must belong to scan file i
    file_ids = [re.search(r"sub-pixar\d+", f).group() for f in data.func]
    assert file_ids == list(pheno["participant_id"]), "Participant table and scan files are not aligned"
    return data, pheno


def compute_motion(confound_files, fd_cutoff=FD_THRESHOLD):
    """Summarise head motion for each participant.

    Framewise displacement (FD; Power et al., 2012) is how far the head moved
    between two consecutive volumes, in mm.
    Returns mean FD and the share of volumes with FD above the cut-off.
    """
    rows = []
    for conf_file in confound_files:
        fd = pd.read_csv(conf_file, sep="\t")["framewise_displacement"].fillna(0)
        rows.append({"mean_fd": fd.mean(), "high_motion_frac": (fd > fd_cutoff).mean()})
    return pd.DataFrame(rows)


def flag_high_motion(motion, fd_threshold=FD_THRESHOLD, frac_threshold=FRAC_THRESHOLD):
    """True for participants who moved too much to keep."""
    return (motion["mean_fd"] > fd_threshold) | (motion["high_motion_frac"] > frac_threshold)


def load_confounds(conf_file, n_compcor=5):
    """Pick the noise signals to remove from one participant's data.

    - 6 head-motion parameters (movement creates fake signal changes)
    - white matter and CSF signal (no neurons there: breathing, heartbeat, drift)
    - the first aCompCor components (main noise patterns; Behzadi et al., 2007)

    We do NOT remove the global (whole-brain) signal, because that can create
    artificial negative correlations.
    """
    conf = pd.read_csv(conf_file, sep="\t")
    motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    tissue = ["white_matter", "csf"]
    compcor = [c for c in conf.columns if c.startswith("a_comp_cor")][:n_compcor]
    cols = [c for c in motion + tissue + compcor if c in conf.columns]
    # The first row has missing values (there is no previous volume), so set them to 0
    return conf[cols].fillna(0).values


def make_masker(atlas_maps, t_r=TR):
    """Tool that averages the voxels of each atlas region into one cleaned signal."""
    return NiftiMapsMasker(
        maps_img=atlas_maps,
        standardize="zscore_sample",  # each region: mean 0, std 1
        detrend=True,                 # remove slow linear drift
        high_pass=0.01,               # remove very slow drift (< 0.01 Hz)
        low_pass=0.1,                 # remove fast noise (> 0.1 Hz: breathing, heartbeat)
        t_r=t_r,
        verbose=0,
    )


def extract_time_series(func_files, conf_files, masker, ids, cache_path=None):
    """Return one (time points x regions) array per participant.

    Extraction takes a while, so the result is saved to `cache_path` and
    loaded from there next time.
    """
    if cache_path and os.path.exists(cache_path):
        saved = np.load(cache_path, allow_pickle=True)
        if list(saved["ids"]) == list(ids):
            print(f"Loaded saved time series from {cache_path}")
            return list(saved["time_series"])
        print("Saved file is for a different sample, extracting again")

    time_series = []
    for i, (func, conf) in enumerate(zip(func_files, conf_files)):
        time_series.append(masker.fit_transform(func, confounds=load_confounds(conf)))
        if (i + 1) % 20 == 0:
            print(f"  done {i + 1}/{len(func_files)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, time_series=np.stack(time_series), ids=np.array(ids))
        print(f"Saved time series to {cache_path}")
    return time_series
