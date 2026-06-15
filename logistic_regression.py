import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plot

from dataclasses import dataclass

from typing import List, Any, Self, Optional, Tuple

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

@dataclass
class WineDataset:

    data: NDArray[np.float64]
    target: NDArray[np.int64]
    feature_names: List[str]
    target_names: List[np.str_]

    @classmethod
    def process(cls) -> Self:
        raw_data: np.ndarray = load_wine()
        return cls(
            data = raw_data.data,
            target = raw_data.target,
            feature_names = raw_data.feature_names,
            target_names = raw_data.target_names
        )

    def split(self, size: float = 0.2, state: int = 42):
        return train_test_split(
            self.data,
            self.target,
            test_size = size,
            random_state = state
        )


class LogisticRegression:

    def __init__(self, x_train: NDArray[np.float64], y_train: NDArray[np.int64]):
        self.__p: Optional[NDArray[np.float64]] = None
        self.learning_rate: float = 0.01
        self.x_train = x_train
        self.y_train = y_train

    def __sigmoid_function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1/(1 + np.exp(-z))
    
    def __cross_entropy(self) -> float:
        """
        p -> probability output of sigmoid function
        y -> train split target names
        """
        return (1/len(self.y)) * (-1 * (np.sum(self.y_train * np.log(self.__p) + (1-self.y_train) * np.log(1-self.__p))))
        
    def __calculate_gradient(self) -> Tuple[NDArray[np.float64], float]:
        sample_number: int = self.x_train.shape[0]
        error: NDArray[np.float64] = self.__p - self.y_train

        weight_gradient: NDArray[np.float64] = (1/sample_number) * self.x_train.T @ error
        bias_gradient: float = (1/sample_number) * np.sum(error)

        return weight_gradient, bias_gradient

    def __gradient_descent(self, iteration_number: int) -> Tuple[NDArray[np.float64], float, List[float]]:
        sample_number, feature_number = self.x_train.shape
        
        weight_array: NDArray[np.float64] = np.zeros(feature_number)
        bias: float = 0.0

        cost_history: List[float] = []

        for i in range(iteration_number):

            z: NDArray[np.float64] = self.x_train @ weight_array + bias
            self.__p = self.__sigmoid_function(z = z)
            weight_gradient, bias_gradient = self.__calculate_gradient()
            weight_array -= self.learning_rate * weight_gradient
            bias -= self.learning_rate * bias_gradient

            cost_history.append(self.__cross_entropy())

        return (weight_array, bias, cost_history)
    
    def fit(self, iteration_number: int = 1000) -> None:
        self.weights, self.bias, self.cost = self.__gradient_descent(iteration_number = iteration_number)