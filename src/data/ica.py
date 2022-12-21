import mne
import numpy as np
from mne.preprocessing import ICA

from settings import EOG_THRESHOLD, EOG_CHANNEL, ECG_THRESHOLD, MUSCLE_THRESHOLD, MONTAGE, N_ICA_COMPONENTS, \
    CHANNEL_NAMES, NON_PHYSIOLOGICAL_CHANNELS, SAMPLING_FREQUENCY, SEED, ECG_CHANNEL, HEOG_THRESHOLD


def filter_ica(run):
    mne_session = mne.io.RawArray(run.session,
                                  mne.create_info(
                                      ch_names=[ch for ch in CHANNEL_NAMES if ch not in NON_PHYSIOLOGICAL_CHANNELS],
                                      sfreq=SAMPLING_FREQUENCY, ch_types="eeg"))

    heog = mne.channels.combine_channels(mne_session, groups=dict(HEOG=[6, 41]),
                                         method=lambda data: np.diff(data, axis=0).squeeze())

    mne_session.filter(1, 50)
    ica = ICA(n_components=N_ICA_COMPONENTS, max_iter='auto', random_state=SEED)
    ica.fit(mne_session, 'eeg')
    mne_session = mne_session.add_channels([heog], force_update_info=True)

    bad_heog, heog_scores = ica.find_bads_eog(mne_session, ch_name="HEOG", measure='correlation',
                                              threshold=HEOG_THRESHOLD)

    bad_eog, eog_scores = ica.find_bads_eog(mne_session, ch_name=EOG_CHANNEL, measure='correlation',
                                            threshold=EOG_THRESHOLD)

    bad_ecg, ecg_scores = ica.find_bads_ecg(mne_session, ch_name=ECG_CHANNEL, method='correlation',
                                            measure='correlation',
                                            threshold=ECG_THRESHOLD)

    mne_session = mne_session.drop_channels("HEOG")

    montage = mne.channels.make_standard_montage(MONTAGE)
    mne_session.set_montage(montage)

    bad_muscle, muscle_scores = ica.find_bads_muscle(mne_session, threshold=MUSCLE_THRESHOLD)
    excludes = bad_eog + bad_heog + bad_ecg + bad_muscle
    print(f"Excluded {len(excludes)} ICs")

    out = ica.apply(mne_session, exclude=excludes)
    return out.get_data()

