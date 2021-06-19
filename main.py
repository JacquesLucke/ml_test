from __future__ import annotations
import numpy as np
import random


class Layer:
    def build(self, shape_in: tuple[int]) -> tuple[int]:
        """
        Returns output shape.
        """
        raise NotImplementedError()

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        """
        Returns output tensor.
        """
        raise NotImplementedError()

    def gradient(
        self, data_in: np.ndarray, diff_out: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        Returns (diff_in, diff_params).
        """
        raise NotImplementedError()

    def apply_gradient(self, diff_params: np.ndarray):
        """
        Add the computed gradient to the parameters of this layer.
        """
        raise NotImplementedError()


class ReLU(Layer):
    def build(self, shape_in: tuple[int]) -> tuple[int]:
        return shape_in

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        data_out = np.maximum(data_in, 0)
        return data_out

    def gradient(
        self, data_in: np.ndarray, diff_out: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        mask = data_in < 0
        diff_in = data_in * mask
        return (diff_in, np.array([]))

    def apply_gradient(self, diff_params: np.ndarray):
        # Nothing to do
        pass


class FullyConnected(Layer):
    def __init__(self, units_out: int):
        self.units_out = units_out

    def build(self, shape_in: tuple[int]) -> tuple[int]:
        assert len(shape_in) == 1
        weights_shape = (self.units_out, shape_in[0])
        self.weights = np.random.default_rng().standard_normal(weights_shape)
        return (self.units_out,)

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        return self.weights @ data_in

    def gradient(
        self, data_in: np.ndarray, diff_out: np.ndarray
    ) -> (np.ndarray, np.ndarray):

        diff_in = self.weights.T @ diff_out
        diff_weights = np.reshape(diff_out, (diff_out.shape[0], 1)) @ np.reshape(
            data_in, (1, data_in.shape[0])
        )
        return (diff_in, diff_weights)

    def apply_gradient(self, diff_params: np.ndarray):
        self.weights += diff_params


class Flatten(Layer):
    def build(self, shape_in: tuple[int]) -> tuple[int]:
        return (np.prod(shape_in),)

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        return data_in.flatten()

    def gradient(
        self, data_in: np.ndarray, diff_out: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        return (diff_out.reshape(data_in.shape), np.array([]))

    def apply_gradient(self, diff_params):
        # Nothing to do.
        pass


class Bias(Layer):
    def build(self, shape_in: tuple[int]) -> tuple[int]:
        self.bias = np.random.default_rng().standard_normal(shape_in)
        return shape_in

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        return data_in + self.bias

    def gradient(
        self, data_in: np.ndarray, diff_out: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        return (diff_out, diff_out.copy())

    def apply_gradient(self, diff_params: np.ndarray):
        self.bias += diff_params


class Model:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def build(self, input_shape: tuple[int]) -> tuple[int]:
        next_shape = input_shape
        for layer in self.layers:
            next_shape = layer.build(next_shape)
        return next_shape

    def compute(self, data_in: np.ndarray) -> np.ndarray:
        data = data_in
        for layer in self.layers:
            data = layer.compute(data)
        return data

    def train_chunk(
        self, data_set_in: list[np.ndarray], data_set_out: list[np.ndarray]
    ):
        assert len(data_set_in) == len(data_set_out)

        gradients = None
        for orig_data_in, data_out in zip(data_set_in, data_set_out):
            data_in = orig_data_in.copy()
            data_ins = []
            for layer in self.layers:
                data_ins.append(data_in)
                data_in = layer.compute(data_in)
            result = data_in
            error = data_out - result
            loss = (error * error).sum()
            print(error)
            print(loss)
            if loss > 10:
                a = 0
            diff_out = error

            local_gradients = []
            for layer, data_in in reversed(list(zip(self.layers, data_ins))):
                diff_out, diff_weights = layer.gradient(data_in, diff_out)
                local_gradients.append(diff_weights)
            local_gradients.reverse()

            if gradients is None:
                gradients = local_gradients
            else:
                for gradient, diff in zip(gradients, local_gradients):
                    gradient += diff

        learning_rate = 0.01 / len(data_set_in)
        for layer, gradient in zip(self.layers, gradients):
            layer.apply_gradient(gradient * learning_rate)

    def train(
        self,
        data_set_in: list[np.ndarray],
        data_set_out: list[np.ndarray],
        *,
        chunk_size,
        chunk_amount
    ):
        assert len(data_set_in) == len(data_set_out)
        all_indices = tuple(range(len(data_set_in)))
        for _ in range(chunk_amount):
            indices = random.choices(all_indices, k=chunk_size)
            samples_in = [data_set_in[i] for i in indices]
            samples_out = [data_set_out[i] for i in indices]
            self.train_chunk(samples_in, samples_out)


model = Model(
    [
        FullyConnected(1),
        Bias(),
    ]
)

model.build((2,))


def generate_data_set(n):
    data_set_in = []
    data_set_out = []

    for _ in range(n):
        x = np.random.default_rng().standard_normal((2,))
        y = (3 * x[0] - 2 * x[1]) - 1
        data_set_in.append(x)
        data_set_out.append(y)

    return data_set_in, data_set_out


train_set_in, train_set_out = generate_data_set(10000)

model.train(train_set_in, train_set_out, chunk_size=50, chunk_amount=1000)

print(model.layers[0].weights)
print(model.layers[1].bias)
# print(model.layers[2].weights)
