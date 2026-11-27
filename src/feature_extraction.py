"""Atlas, connectivity matrices and network-level summaries."""

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure


def load_atlas():
    """MSDL atlas: 39 regions in 17 networks."""
    msdl = datasets.fetch_atlas_msdl(verbose=0)
    return msdl.maps, np.array(msdl.labels), np.array(msdl.networks), np.array(msdl.region_coords)


def connectivity_matrices(time_series, kind="correlation"):
    return ConnectivityMeasure(kind=kind, standardize="zscore_sample").fit_transform(time_series)


def fisher_z(corr):
    return np.arctanh(np.clip(corr, -0.999, 0.999))


def group_mean(corr_mats, mask):
    """Mean correlation matrix of a group, averaged in Fisher-z space."""
    return np.tanh(fisher_z(corr_mats[mask]).mean(axis=0))


def network_summary(mat, networks, net_names):
    """Mean connectivity within each network (diagonal) and between each pair of networks."""
    out = pd.DataFrame(index=net_names, columns=net_names, dtype=float)
    for a in net_names:
        for b in net_names:
            ia, ib = np.where(networks == a)[0], np.where(networks == b)[0]
            block = mat[np.ix_(ia, ib)]
            if a == b:
                block = block[~np.eye(len(ia), dtype=bool)]
            out.loc[a, b] = block.mean() if block.size else np.nan
    return out


def within_between(z_mats, networks):
    """Per participant: mean within-network and mean between-network connectivity."""
    same_net = networks[:, None] == networks[None, :]
    off_diag = ~np.eye(len(networks), dtype=bool)
    return z_mats[:, same_net & off_diag].mean(axis=1), z_mats[:, ~same_net].mean(axis=1)


def edge_table(labels, networks, coords):
    """One row per region pair, in nilearn's vectorized order (lower triangle)."""
    rows, cols = np.tril_indices(len(labels), k=-1)
    return pd.DataFrame({
        "region_A": labels[rows], "network_A": networks[rows],
        "region_B": labels[cols], "network_B": networks[cols],
        "same_network": networks[rows] == networks[cols],
        "distance_mm": np.linalg.norm(coords[rows] - coords[cols], axis=1),
    })


def vector_to_matrix(values, n_regions):
    rows, cols = np.tril_indices(n_regions, k=-1)
    mat = np.zeros((n_regions, n_regions))
    mat[rows, cols] = values
    return mat + mat.T


def vectorize(time_series, kind="correlation"):
    measure = ConnectivityMeasure(kind=kind, vectorize=True, discard_diagonal=True,
                                  standardize="zscore_sample")
    return measure.fit_transform(time_series)
