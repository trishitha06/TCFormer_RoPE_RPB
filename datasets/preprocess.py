import numpy as np
from scipy.signal import butter, filtfilt


############################################################
# Bandpass Filter
############################################################

def bandpass_filter(
    data,
    lowcut=2,
    highcut=40,
    fs=250,
    order=4
):

    nyquist = 0.5 * fs

    low = lowcut / nyquist

    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered = filtfilt(
        b,
        a,
        data,
        axis=-1
    )

    return filtered


############################################################
# Exponential Moving Standardization
############################################################

def exponential_moving_standardize(
    data,
    factor_new=0.001,
    eps=1e-4
):

    mean = np.zeros_like(data)

    var = np.zeros_like(data)

    standardized = np.zeros_like(data)

    mean[..., 0] = data[..., 0]

    var[..., 0] = 0.0

    standardized[..., 0] = 0.0

    for t in range(1, data.shape[-1]):

        mean[..., t] = (

            factor_new * data[..., t]

            + (1 - factor_new) * mean[..., t - 1]

        )

        var[..., t] = (

            factor_new * (data[..., t] - mean[..., t]) ** 2

            + (1 - factor_new) * var[..., t - 1]

        )

        standardized[..., t] = (

            data[..., t] - mean[..., t]

        ) / np.maximum(

            np.sqrt(var[..., t]),

            eps

        )

    return standardized


############################################################
# Complete Preprocessing
############################################################

def preprocess_trial(trial):

    trial = bandpass_filter(

        trial,

        lowcut=2,

        highcut=40,

        fs=250,

        order=4

    )

    trial = exponential_moving_standardize(

        trial

    )

    trial = np.nan_to_num(

        trial,

        nan=0.0,

        posinf=0.0,

        neginf=0.0

    )

    return trial.astype(np.float32)