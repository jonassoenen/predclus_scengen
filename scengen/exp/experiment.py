import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from scengen.evaluation import calculate_energy_scores_for_index_samples


class ComparisonExperiment:
    def __init__(self, datasets=None, methods=None, folds=None, result_path=None, nb_of_samples= 250, crossval=True):
        self.datasets = datasets if datasets else dict()
        self.methods = methods if methods else dict()
        self.folds = folds if folds else dict()
        self.result_path = result_path
        self.nb_of_samples = nb_of_samples
        self.crossval = crossval

    def add_dataset(self, dataset_name, attributes, timeseries, folds):
        self.datasets[dataset_name] = (attributes, timeseries)
        self.folds[dataset_name] = folds
        return self

    def add_method(self, method_name, generator):
        self.methods[method_name] = generator
        return self

    def add_datasets(self, **kwargs):
        for dataset_name, args in kwargs.items():
            self.add_dataset(dataset_name, *args)
        return self
    def add_methods(self, **kwargs):
        for method_name, generator in kwargs.items():
            self.add_method(method_name, generator)
        return self

    def set_result_path(self, path):
        self.result_path = path
        return self

    def train_test_sets(self, folds):
        if len(folds) == 2:
            # single train test split
            yield folds
        elif not self.crossval:
            train_set = np.concatenate(folds[:-1], axis=0)
            test_set = folds[-1]
            yield train_set, test_set
        else:
            for test_idx in range(0, len(folds)):
                train_set = np.concatenate(folds[:test_idx] + folds[test_idx + 1:], axis=0)
                test_set = folds[test_idx]
                yield train_set, test_set

    def execute(self):
        self.result_path.mkdir(parents=True, exist_ok=True)
        keys = []
        all_energy_scores = []
        all_timings = []
        with tqdm(self.methods.items(), desc="Methods") as method_iter:
            for method_name, generator in method_iter:
                method_iter.set_postfix(method = method_name)
                with tqdm(self.datasets.items(), desc= 'Datasets', leave=False) as dataset_iter:
                    for dataset_name, dataset in dataset_iter:
                        dataset_iter.set_postfix(dataset = dataset_name)
                        folds = self.folds[dataset_name]
                        energy_scores, timings = self.execute_single_method(method_name, generator, dataset_name,
                                                                            dataset, folds)
                        all_energy_scores.append(energy_scores)
                        all_timings.append(timings)
                        keys.append((method_name, dataset_name))
        energy_scores = pd.concat(all_energy_scores, axis=1, keys=keys)
        timings = pd.concat(all_timings, axis=1, keys=keys).T
        return energy_scores, timings

    def execute_single_method(self, method_name, method, dataset_name, dataset, folds):
        single_result_path = self.result_path / f"{method_name}_{dataset_name}.pkl"

        if single_result_path.exists():
            # instead of calculating load from disk
            with single_result_path.open(mode='rb') as file:
                return pickle.load(file)

        with tqdm(self.train_test_sets(folds)) as fold_iter:
            energy_scores_per_fold = []
            training_time = 0
            predict_time = 0
            eval_time = 0

            for train_set, test_set in fold_iter:
                attributes, timeseries = dataset

                # fit the method on training set
                start_time = time.time()
                method.fit(attributes[train_set], timeseries[train_set])
                training_time += time.time() - start_time

                # generate predictions for test set
                start_time = time.time()
                predictions = method.generate(attributes[test_set], self.nb_of_samples)
                predict_time += time.time() - start_time

                # compare predictions to ground truth
                start_time = time.time()
                energy_scores = calculate_energy_scores_for_index_samples(timeseries[train_set], predictions,
                                                                          timeseries[test_set])
                eval_time += time.time() - start_time

                # save the result
                energy_scores_per_fold.append(energy_scores)

        if len(energy_scores_per_fold) == 1:
            # only one test set
            energy_scores = pd.Series(energy_scores_per_fold[0], index=test_set)
        else:
            # multiple test sets (cross validation)
            energy_scores = pd.Series(np.concatenate(energy_scores_per_fold, axis=0))

        timings = pd.Series([training_time, predict_time, eval_time],
                            index=['training_time', 'predict_time', 'eval_time'])
        result = energy_scores, timings

        # cache result on disk
        with single_result_path.open(mode='wb') as file:
            pickle.dump(result, file)

        return result
