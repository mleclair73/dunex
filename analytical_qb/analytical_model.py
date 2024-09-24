import numpy as np

# --Waves-- #


def TG89_W(H, H_rms, h, gamma=0.35, n=4):
    # Thorton, Guza ('89) breaking weight function
    return np.minimum(
        (H_rms / gamma / h) ** n * (1 - np.exp(-((H / gamma / h) ** 2))), 1
    )


def rayleight_distribution(H, H_rms):
    # This is from Thorton, Guza ('89) #16, citing [Cartwright & Longuet-Higgins, 1956]
    return 2 * H / H_rms**2 * np.exp(-((H / H_rms) ** 2))


def calculate_H_max(H_rms, N):
    # This is from Thorton, Guza ('89) #16, citing [Cartwright & Longuet-Higgins, 1956]
    return (np.log(N) ** 0.5 + 0.2886 * np.log(N) ** -0.5) * H_rms


def H_s_to_H_rms(H_s):
    return H_s / 2 * np.sqrt(2)

def shoaling_coefficient(profile):
    return np.nan_to_num(np.sqrt(profile[1:] / profile[:-1]), nan=1)


# --Shoreline Utilities-- #


def shoreline(bathymetry):
    # Returns the leftmost non-nan value along the first axis
    indices = np.isnan(bathymetry) * np.arange(0, bathymetry.shape[1])
    return np.argmax(indices, axis=1)


def shoreline_align(data, bathymetry, sl=None):
    # Align all data to the shoreline
    if sl is None:
        sl = shoreline(bathymetry)
    aligned = np.zeros_like(data)
    for i, start in enumerate(sl):
        aligned[i, 0 : aligned.shape[1] - start] = data[i, start:]
    return aligned

def shoreline_average(data, bathymetry):
    # Calculate a shoreline adjusted cross-shore average 
    sl = shoreline(bathymetry)
    aligned = shoreline_align(data, bathymetry, sl)
    average = np.zeros(data.shape[1]) * np.nan
    average[int(sl.mean()) :] = aligned.mean(0)[: -int(sl.mean())]
    return average
