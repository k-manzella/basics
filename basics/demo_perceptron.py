"""This will test how well I wrote dimension handling for the 
matrix multiplication function; lots of multiplying vectors V*M"""

from typing import List

import basics

class SinglePerceptronClassifier:
    def __init__(self, train_test_split_p: float=0.2):
        self.train_test_split_p = train_test_split_p
        self.dimensions = None
        self.n_target = None
        self.y_input = None
        self.x_input = None
        self.weights = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None

    def __init_weights(self, n_weights, method="zeros"):
        """
        param method: one of ["zeros", "normal"]
        """
        def get_zeros(n):
            return [0 for i in range(n)]

        func_dict = {
            "zeros": get_zeros,
            "normal": basics.draw_random_normal
        }
        self.weights = func_dict[method](n_weights)

    def __preprocess_target(self, Y):
        y_out = None
        if len(Y) == 1:
            Y = Y[0]
        if len(Y) == 1:
            raise ValueError(f"Singular target vector Y passed: {Y}")
        
        self.n_target = len(Y)
        self.y_input = Y

    def __preprocess_data(self, X):
        if len(X) != self.n_target:
            X = basics.transpose(X)
        if len(X) != self.n_target:
            raise ValueError(f"Dimension mismatch: X length {len(X)}, Y length {self.n_target}")
        self.dimensions = (self.n_target, len(X[0]))

    def train(self, X, Y):
        self.__preprocess_target(Y)
        self.__preprocess_data(X)
