"""
Feature extraction: turn region signals into connectivity features.

Connectivity = how strongly two brain regions rise and fall together.
Used by notebooks 02, 03 and 04.
"""

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure


def load_atlas():
    """MSDL atlas: 39 regions grouped into 17 known networks (Varoquaux et al., 2011).

    Returns the atlas maps plus region names, network names and (x, y, z) coordinates.
    """
    msdl = datasets.fetch_atlas_msdl(verbose=0)
    labels = np.array(msdl.labels)
    networks = np.array(msdl.networks)
    coords = np.array(msdl.region_coords)
    return msdl.maps, labels, networks, coords


def connectivity_matrices(time_series, kind="correlation"):
    """One (regions x regions) matrix per participant."""
    measure = ConnectivityMeasure(kind=kind, standardize="zscore_sample")
    return measure.fit_transform(time_series)


def fisher_z(corr):
    """Fisher z-transform. Correlations are bounded at +/-1, so they can't be averaged
    directly; arctanh puts them on an unbounded scale where averaging is valid."""
    return np.arctanh(np.clip(corr, -0.999, 0.999))


def group_mean(corr_mats, mask):
    """Average correlation matrix of a group (averaged in z-space, converted back)."""
    return np.tanh(fisher_z(corr_mats[mask]).mean(axis=0))


def network_summary(mat, networks, net_names):
    """Average connectivity within each network (diagonal) and between each pair of networks."""
    out = pd.DataFrame(index=net_names, columns=net_names, dtype=float)
    for a in net_names:
        for b in net_names:
            ia, ib = np.where(networks == a)[0], np.where(networks == b)[0]
            block = mat[np.ix_(ia, ib)]
            if a == b:  # within a network: leave out self-connections
                block = block[~np.eye(len(ia), dtype=bool)]
            out.loc[a, b] = block.mean() if block.size else np.nan
    return out


def within_between(z_mats, networks):
    """Per participant: mean connectivity within networks and between networks."""
    same_net = networks[:, None] == networks[None, :]
    off_diag = ~np.eye(len(networks), dtype=bool)
    within = z_mats[:, same_net & off_diag].mean(axis=1)
    between = z_mats[:, ~same_net].mean(axis=1)
    return within, between


def edge_table(labels, networks, coords):
    """One row per connection (region pair), in the same order nilearn uses when it
    flattens a matrix into a vector (lower triangle, diagonal left out)."""
    rows, cols = np.tril_indices(len(labels), k=-1)
    return pd.DataFrame({
        "region_A": labels[rows], "network_A": networks[rows],
        "region_B": labels[cols], "network_B": networks[cols],
        "same_network": networks[rows] == networks[cols],
        "distance_mm": np.linalg.norm(coords[rows] - coords[cols], axis=1),
    })


def vector_to_matrix(values, n_regions):
    """Put one value per connection back into a symmetric (regions x regions) matrix."""
    rows, cols = np.tril_indices(n_regions, k=-1)
    mat = np.zeros((n_regions, n_regions))
    mat[rows, cols] = values
    return mat + mat.T


def vectorize(time_series, kind="correlation"):
    """Flatten each participant's matrix into a vector of connections (741 for 39 regions)."""
    measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True,
                                  standardize="zscore_sample")
    return measure.fit_transform(time_series)
