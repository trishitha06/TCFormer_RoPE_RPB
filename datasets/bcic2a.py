import os

import mne
import numpy as np
import torch

from torch.utils.data import Dataset

from datasets.preprocess import preprocess_trial
from datasets.augmentation import EEGAugmentation


EVENTS = {
    "769": 0,   # Left Hand
    "770": 1,   # Right Hand
    "771": 2,   # Feet
    "772": 3    # Tongue
}


SELECTED_CHANNELS = [
    "C3", "Cz", "C4",
    "FC1", "FC2",
    "FC3", "FC4",
    "FC5", "FC6",
    "CP1", "CP2",
    "CP3", "CP4",
    "CP5", "CP6",
    "C1", "C2",
    "C5", "C6",
    "FCz",
    "CPz",
    "Pz"
]


class BCIC2aDataset(Dataset):

    def __init__(
        self,
        root,
        subjects,
        train=True
    ):
    

        self.root = root
        self.subjects = subjects
        self.train = train

        self.augment = EEGAugmentation()

        self.data = []
        self.labels = []

        self.load_dataset()

    def load_dataset(self):

        for subject in self.subjects:

            if self.train:
                filename = f"A{subject:02d}T.gdf"
            else:
                filename = f"A{subject:02d}E.gdf"

            filepath = os.path.join(
                self.root,
                filename
            )

            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"{filepath} not found."
                )

            print(f"Loading {filename}")

            self.load_subject(filepath)

        self.data = np.asarray(
            self.data,
            dtype=np.float32
        )

        self.labels = np.asarray(
            self.labels,
            dtype=np.int64
        )


    def load_subject(self, filepath):

        raw = mne.io.read_raw_gdf(
            filepath,
            preload=True,
            verbose=False
        )

        print("\nOriginal Channels:")
        print(raw.ch_names)
        # Keep the first 22 EEG channels.
        # The last 3 channels are EOG.

        eeg_channels = raw.ch_names[:22]

        print("\nUsing EEG Channels:")
        print(eeg_channels)

        raw.pick(eeg_channels)
        events, event_dict = mne.events_from_annotations(raw)

        event_mapping = {}

        for event_name in EVENTS.keys():

            if event_name in event_dict:

                event_mapping[event_name] = event_dict[event_name]

        if len(event_mapping) == 0:

            raise RuntimeError(
                f"No motor imagery events found in {filepath}"
            )

        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=event_mapping,
            tmin=2.0,
            tmax=6.0,
            baseline=None,
            preload=True,
            verbose=False
        )

        X = epochs.get_data(copy=True)

        print("Epoch Shape:", X.shape)

        y = epochs.events[:, -1]

        reverse_mapping = {
            value: EVENTS[key]
            for key, value in event_mapping.items()
        }

        for trial, label in zip(X, y):

            trial = preprocess_trial(trial)

            self.data.append(trial)

            self.labels.append(
                reverse_mapping[label]
            )
            
    def __len__(self):

        return len(self.data)

    ############################################################

    def __getitem__(self, index):

        trial = self.data[index].copy()

        label = self.labels[index]

        if self.train:

            trial = self.augment(trial)

        trial = torch.from_numpy(
            trial
        ).float()

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return trial, label
