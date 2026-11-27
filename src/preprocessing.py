"""Loading the data, motion QC and region time-series extraction."""

import os
import re
import time

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker

TR = 2.0  # from the paper (the file header says 1.0)

# exclusion rules, fixed before running the analysis
FD_THRESHOLD = 0.5    # mm
FRAC_THRESHOLD = 0.2  # max share of volumes with FD > 0.5 mm


def load_dataset(n_subjects=None, max_tries=10):
    """Returns the nilearn dataset and the participants table (same order as data.func)."""
    # OSF times out sometimes; finished files stay on disk, so retrying resumes the download
    for attempt in range(1, max_tries + 1):
        try:
            data = datasets.fetch_development_fmri(n_subjects=n_subjects, verbose=0)
            break
        except Exception as err:
            if attempt == max_tries:
                raise
            print(f"download failed ({type(err).__name__}), retrying {attempt}/{max_tries - 1}")
            time.sleep(5)

    pheno = pd.DataFrame(data.phenotypic).reset_index(drop=True)
    file_ids = [re.search(r"sub-pixar\d+", f).group() for f in data.func]
    assert file_ids == list(pheno["participant_id"]), "participants table doesn't match scan files"
    return data, pheno


def compute_motion(confound_files, fd_cutoff=FD_THRESHOLD):
    """Mean framewise displacement and share of high-motion volumes, per participant."""
    rows = []
    for f in confound_files:
        fd = pd.read_csv(f, sep="\t")["framewise_displacement"].fillna(0)
        rows.append({"mean_fd": fd.mean(), "high_motion_frac": (fd > fd_cutoff).mean()})
    return pd.DataFrame(rows)


def flag_high_motion(motion, fd_threshold=FD_THRESHOLD, frac_threshold=FRAC_THRESHOLD):
    return (motion["mean_fd"] > fd_threshold) | (motion["high_motion_frac"] > frac_threshold)


def load_confounds(conf_file, n_compcor=5):
    """6 motion params + white matter + CSF + first aCompCor components. No global signal."""
    conf = pd.read_csv(conf_file, sep="\t")
    motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    tissue = ["white_matter", "csf"]
    compcor = [c for c in conf.columns if c.startswith("a_comp_cor")][:n_compcor]
    cols = [c for c in motion + tissue + compcor if c in conf.columns]
    return conf[cols].fillna(0).values  # first row is NaN for motion derivatives


def make_masker(atlas_maps, t_r=TR):
    return NiftiMapsMasker(
        maps_img=atlas_maps,
        standardize="zscore_sample",
        detrend=True,
        high_pass=0.01,
        low_pass=0.1,
        t_r=t_r,
        verbose=0,
    )


def extract_time_series(func_files, conf_files, masker, ids, cache_path=None):
    """One (time points x regions) array per participant. Cached to cache_path."""
    if cache_path and os.path.exists(cache_path):
        saved = np.load(cache_path, allow_pickle=True)
        if list(saved["ids"]) == list(ids):
            print("loaded", cache_path)
            return list(saved["time_series"])

    time_series = []
    for i, (func, conf) in enumerate(zip(func_files, conf_files)):
        time_series.append(masker.fit_transform(func, confounds=load_confounds(conf)))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(func_files)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, time_series=np.stack(time_series), ids=np.array(ids))
    return time_series


def prepare_data(data_dir, atlas_maps):
    """Clean participants table + region time series (the outputs of notebook 01).

    Rebuilds them if they're missing, e.g. on Colab where each notebook runs on its own machine.
    """
    csv_path = f"{data_dir}/participants_clean.csv"
    ts_path = f"{data_dir}/time_series_msdl.npz"

    if not os.path.exists(csv_path):
        print("notebook 01 outputs not found, rebuilding (download + extraction, a few minutes)")
        data, pheno = load_dataset()
        pheno = pd.concat([pheno, compute_motion(data.confounds)], axis=1)
        pheno["exclude"] = flag_high_motion(pheno)
        keep = np.where(~pheno.exclude)[0]
        clean = pheno.iloc[keep].reset_index(drop=True)
        extract_time_series([data.func[i] for i in keep], [data.confounds[i] for i in keep],
                            make_masker(atlas_maps), ids=clean.participant_id.values,
                            cache_path=ts_path)
        clean.to_csv(csv_path, index=False)

    pheno = pd.read_csv(csv_path)
    saved = np.load(ts_path, allow_pickle=True)
    assert list(saved["ids"]) == list(pheno["participant_id"]), "delete data/ and rerun"
    return pheno, list(saved["time_series"])
