import numpy as np
import torch


def _state_to_features(psi: np.ndarray) -> torch.Tensor:
    """
    psi from your env is complex and typically shaped (D,1).
    Convert to real feature vector [Re(psi_flat), Im(psi_flat)] of length 2D.
    """
    psi = np.asarray(psi).reshape(-1)  # flatten (D,1) -> (D,)
    feats = np.concatenate([np.real(psi), np.imag(psi)], axis=0).astype(np.float32)
    return torch.from_numpy(feats)


def _srv_to_features(srv) -> torch.Tensor:
    return torch.tensor(list(srv), dtype=torch.float32)


def infer_valid_srvs_for_env(env):
    """
    Enumerate all SRVs in {1..d}^n and keep those passing is_valid_srv(srv, d).
    """
    d = int(env.dim)
    n = int(env.num_ions)

    valid = []
    for idx in np.ndindex(*([d] * n)):
        srv = [i + 1 for i in idx]
        ok, _ = is_valid_srv(srv, d=d)  # uses your function
        if ok:
            valid.append(srv)

    if len(valid) == 0:
        raise RuntimeError("No valid SRVs found. Check is_valid_srv / env.dim / env.num_ions.")
    return valid
