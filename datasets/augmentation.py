import numpy as np


class EEGAugmentation:

    def __init__(
        self,
        noise_std=0.01,
        scale_range=(0.8, 1.2),
        shift_range=15,
        crop_max=20
    ):

        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.crop_max = crop_max

    ############################################################

    def gaussian_noise(self, x):

        noise = np.random.normal(
            0,
            self.noise_std,
            x.shape
        )

        return x + noise

    ############################################################

    def random_scaling(self, x):

        scale = np.random.uniform(
            self.scale_range[0],
            self.scale_range[1]
        )

        return x * scale

    ############################################################

    def random_shift(self, x):

        shift = np.random.randint(
            -self.shift_range,
            self.shift_range + 1
        )

        return np.roll(
            x,
            shift,
            axis=-1
        )

    ############################################################

    def random_crop(self, x):

        crop = np.random.randint(
            0,
            self.crop_max + 1
        )

        if crop == 0:
            return x

        cropped = x[:, crop:]

        pad = np.zeros(
            (cropped.shape[0], crop),
            dtype=x.dtype
        )

        cropped = np.concatenate(
            [cropped, pad],
            axis=1
        )

        return cropped

    ############################################################

    def __call__(self, x):

        if np.random.rand() < 0.5:
            x = self.gaussian_noise(x)

        if np.random.rand() < 0.5:
            x = self.random_scaling(x)

        if np.random.rand() < 0.5:
            x = self.random_shift(x)

        if np.random.rand() < 0.3:
            x = self.random_crop(x)

        return x.astype(np.float32)
