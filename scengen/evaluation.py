import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances

def calculate_energy_scores_for_index_samples(train_timeseries, all_samples, correct_profiles):
    energy_scores = np.zeros(all_samples.shape[0])
    for i in range(all_samples.shape[0]):
        samples_idx = all_samples[i]
        correct_profile = correct_profiles[i]
        samples_ts = train_timeseries[samples_idx]
        energy_scores[i] = calculate_energy_score(samples_ts, correct_profile)
    return energy_scores

def calculate_energy_score(samples, correct_profile):
    """
    samples: 2d numpy array shape (#samples, #dim) , rows are predicted samples
    correct_profile: 1d numpy array shape (#dim), the ground truth sample
    """
    # sum of distances between ground truth and each sample
    first_term = np.sum(
        euclidean_distances(samples, correct_profile.reshape((1, -1))).squeeze()
    )

    # sum of pairwise distances between samples
    second_term = pairwise_distances(samples).sum(axis=None)

    nb_of_samples = samples.shape[0]
    return 1 / nb_of_samples * first_term - 0.5 * 1 / nb_of_samples ** 2 * second_term

