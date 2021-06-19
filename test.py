import numpy as np

weights = np.array(
    [
        [[1, 2], [3, 4]],
        [[-1, -2], [-3, -4]],
        [[0, 3], [-1, 2]],
    ]
)

data_in = np.array(
    [
        [1, 2],
        [3, 4],
    ]
)

print(weights.shape)
print(weights)
print(data_in.shape)
print(data_in)
print(np.tensordot(weights, data_in, 2))

expected_result = np.array(
    [
        1 + 4 + 9 + 16,  # 30
        -1 - 4 - 9 - 16,  # -30
        0 + 6 - 3 + 8,  # 11
    ]
)

factor = np.array([10, 100, 1])
print(np.tensordot(factor, weights, 1))
# factor = np.broadcast_to(factor, (2, 2, 3))
# print(factor)
# print(factor * weights)
