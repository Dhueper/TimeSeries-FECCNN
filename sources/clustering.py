from vmdpy import VMD 
import ewtpy

def VMD_clustering(t, X, K_modes):
    """Computes the VMD (Variational Mode Decomposition) of
    the time series.

    Intent(in): X (numpy.array), time series.

    Returns: u (numpy.array), decomposed time series.
    """

    alpha = 1000       # moderate bandwidth constraint  
    tau = t[1] - t[0]             # noise-tolerance (no strict fidelity enforcement) 
    # tau = 0 
    K = K_modes              # Number of modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  

 
    u, u_hat, omega = VMD(X, alpha, tau, K, DC, init, tol)

    return u

def EWT_clustering(X, K_modes):
    """Computes the EWT (Empirical Wavelet Transform) of
    the time series.

    Intent(in): X (numpy.array), time series.

    Returns: ewt (numpy.array), decomposed time series.
    """

    ewt,  mfb ,boundaries = ewtpy.EWT1D(X, N = K_modes)

    return ewt
