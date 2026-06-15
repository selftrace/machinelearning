from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from numpy import ndarray
from dataclasses import dataclass
from typing import Any, Self, List

import matplotlib.pyplot as plot


@dataclass
class BreastCancerData:

    data: ndarray
    target: ndarray
    features_names: ndarray
    target_names: ndarray

    @classmethod
    def process(cls) -> Self:
        raw_data: Any = load_breast_cancer()
        return cls(
            data = raw_data.data,
            target = raw_data.target,
            features_names = raw_data.feature_names,
            target_names = raw_data.target_names
        )
    
    def split(self, size: float = 0.2, state: int = 42) -> List[Any]:
        return train_test_split(
            self.data,
            self.target,
            test_size = size,
            random_state = state
        )


class Training:
    def __init__(self, feature_train: List[Any], feature_test: List[Any], labels_train: List[Any], labels_test: List[Any], raw_data: ndarray):
        self.gnb_model = GaussianNB()
        self.x_test = feature_test
        self.x_train = feature_train
        self.y_test = labels_test
        self.y_train = labels_train
        self.data = raw_data

    
    def train(self) -> None:
        self.gnb_model.fit(self.x_train, self.y_train)
        self.labels_all = self.gnb_model.predict(self.data)

    def plot_data(self, data: ndarray, target: ndarray, feature_name_1: str, feature_name_2: str, accuracy: float) -> None:
        plot.figure(figsize=(10, 10))
        plot.scatter(
            x=data[:, 0],
            y=data[:, 1],
            c=target,
            cmap='coolwarm',
            alpha=0.5,
            edgecolors='k'
        )
        plot.xlabel(feature_name_1)
        plot.ylabel(feature_name_2)
        plot.title(f"Breast Cancer Prediction: {accuracy}")
        plot.colorbar(label="diagnosis")
        plot.show()
    
    def plot_both(self, data: ndarray, target: ndarray) -> None:
        
        predictions: List[bool] = (self.labels_all == target)
        ratio: ndarray = data[:, 0] / data[:, 1]
        figure, axes = plot.subplots(figsize=(20, 10))
        
        axes.scatter(
            x=ratio[predictions],
            y=data[predictions, 0],
            c=target[predictions],
            cmap='coolwarm',
            s=40,
            alpha=0.5,
            edgecolors='k',
            label="correct"
        )

        axes.scatter(
            x=ratio[~predictions],
            y=data[~predictions, 0],
            c='purple',
            marker='X',
            s=100,
            edgecolors='k',
            label=f"Errors: {(~predictions).sum()}"
        )

        axes.set_xlabel("Ratio of mean radius/mean texture")
        axes.set_ylabel("Mean radius")
        axes.set_title("Feature ratio vs errors")
        axes.legend()
        axes.grid(visible=True, alpha=0.3)
        plot.show()


if __name__ == "__main__":
    """
    x -> features
    y -> labels
    """
    raw_data: BreastCancerData = BreastCancerData.process()
    x_train, x_test, y_train, y_test = raw_data.split()

    testing_instance = Training(x_train, x_test, y_train, y_test, raw_data=raw_data.data)
    testing_instance.train()


    """testing_instance.plot_data(
        data=raw_data.data,
        target=raw_data.target,
        feature_name_1=raw_data.features_names[0],
        feature_name_2=raw_data.features_names[1],
        accuracy=f"{(testing_instance.gnb_model.score(x_test, y_test))*100:.2f}%"
    )"""
    

    testing_instance.plot_both(data=raw_data.data, target=raw_data.target)
