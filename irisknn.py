import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plot

from collections import Counter
from typing import List, Any, Self

from dataclasses import dataclass
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

@dataclass
class IrisDataset:

    data: ndarray
    target: ndarray
    feature_names: List
    target_names: ndarray

    @classmethod
    def process(cls) -> Self:
        raw_data = load_iris()
        return cls(
            data = raw_data.data,
            target = raw_data.target,
            feature_names = raw_data.feature_names,
            target_names = raw_data.target_names
        )

    def split(self, size: float = 0.2, state: int = 42) -> List[Any]:
        return train_test_split(
            self.data,
            self.target,
            test_size =  size,
            random_state = state
        )

class KNN:

    @classmethod
    def minkowski_distance(cls, point_1, point_2, p: int = 2) -> Any:
        return (np.sum(np.abs(np.array(point_1) - np.array(point_2))**p))**(1/p)

    @classmethod
    def predict(cls, x_train: ndarray, y_train: ndarray, point, k: int) -> int:
        distance_list: List[Any] = []
        for i in range(len(x_train)):
            distance_list.append((y_train[i], cls.minkowski_distance(point, x_train[i])))
        distance_list.sort(key=lambda x: x[1])
        k_labels = [target for target, distance in distance_list[:k]]
        return Counter(k_labels).most_common(1)[0][0]
    
    @classmethod
    def k_fold_cv(cls, data: ndarray, targets: ndarray, k_val: int, k_fold: int) -> float:
        acc_score: List[float] = []
        fold_size: int = len(data) // k_fold

        for fold in range(k_fold):
            start_range: int = fold * fold_size
            end_range: int = start_range + fold_size

            x_test: ndarray = data[start_range:end_range]
            y_test: ndarray = targets[start_range:end_range]

            x_train: ndarray = np.concatenate([data[:start_range], data[end_range:]])
            y_train = np.concatenate([targets[:start_range], targets[end_range:]])

            correct_points: int = 0
            for i in range(len(x_test)):
                prediction = cls.predict(x_train=x_train, y_train=y_train, point=x_test[i], k=k_val)#
                if prediction == y_test[i]:
                    correct_points += 1
            
            acc_score.append(correct_points / len(x_test))
        
        return sum(acc_score) / len(acc_score)
    
    @classmethod
    def plot_k_fold(cls, k_vals: List[int], accuracies: List[float]) :
        plot.figure(figsize=(25, 15))
        plot.plot(k_vals, accuracies, marker = 'x', linewidth = 2, markersize = 6)
        plot.xlabel("k", fontsize = 10)
        plot.ylabel("k fold CV accuracy", fontsize = 10)
        plot.title("kNN performance vs k Value for Iris dataset", fontsize = 13)
        plot.grid(True, alpha=0.2)
        plot.xticks(k_vals)
        plot.show()

    @classmethod
    def plot_predictions(cls, raw_data, x_train, y_train, x_test, y_test, k):
        prediction_list: np.array = np.array([KNN.predict(x_train=x_train, y_train=y_train, point=point, k=k) for point in x_test])
        plot.figure(figsize=(25, 15))
        plot.subplot(1, 2, 1)
        for i, species in enumerate(raw_data.target_names):
            mask = y_test == i
            plot.scatter(x_test[mask, 0], x_test[mask, 1],
                    label=species, s=100, alpha=0.7)
        plot.xlabel(raw_data.feature_names[0])
        plot.ylabel(raw_data.feature_names[1])
        plot.title('Actual Labels')
        plot.legend()
        plot.grid(True, alpha=0.3)
        
        plot.subplot(1, 2, 2)
        for i, species in enumerate(raw_data.target_names):
            mask = prediction_list == i
            plot.scatter(
                x_test[mask, 0], 
                x_test[mask, 1],
                label=species, 
                s=100, 
                alpha=0.7)
        
        misclassified = prediction_list != y_test
        if np.any(misclassified):
            plot.scatter(
                x_test[misclassified, 0], 
                x_test[misclassified, 1],
                marker='x',
                s=200, 
                c='red', 
                linewidths=3, 
                label='Misclassified')
        
        plot.xlabel(raw_data.feature_names[0])
        plot.ylabel(raw_data.feature_names[1])
        plot.title(f'Predicted Labels (k={k})')
        plot.legend()
        plot.grid(True, alpha=0.3)
        
        plot.tight_layout()
        plot.show()



if __name__ == "__main__":
    raw_data: IrisDataset = IrisDataset.process()
    x_train, x_test, y_train, y_test = raw_data.split()

    k_vals: List[int] = [3, 5, 7, 9, 11, 13, 15]
    accuracies: List[float] = [KNN.k_fold_cv(raw_data.data, raw_data.target, k, 5) for k in k_vals]

    #KNN.plot_k_fold(k_vals=k_vals, accuracies=accuracies)
    KNN.plot_predictions(raw_data, x_train, y_train, x_test, y_test, 7)
