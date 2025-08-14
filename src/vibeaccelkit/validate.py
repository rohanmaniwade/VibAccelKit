import numpy as np

def validate_srs_ers(srs_pos, ers, factor: float = 2.0):
    """
    Returns (is_valid, mask_violations) where mask_violations[i] is True
    when ERS[i] > factor * SRS_pos[i].
    """
    srs_pos = np.asarray(srs_pos, float)
    ers = np.asarray(ers, float)
    if srs_pos.shape != ers.shape:
        raise ValueError("SRS+ and ERS must have the same shape")
    violations = ers > (factor * srs_pos + 1e-30)
    return not np.any(violations), violations

def validate_fds_meets_target(fds_target, fds_candidate, tol_db: float = 0.0):
    """
    Check that synthesized/accelerated FDS meets or exceeds target/composite FDS.
    tol_db <= 0 enforces pointwise >= ; positive tol_db allows small shortfalls.
    """
    eps = 1e-30
    ratio = (np.asarray(fds_candidate)+eps) / (np.asarray(fds_target)+eps)
    if tol_db <= 0:
        return bool(np.all(ratio >= 1.0))
    return bool(np.all(10.0*np.log10(ratio) >= -abs(tol_db)))
