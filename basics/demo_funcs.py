import numpy as np

import basics


def demo_matrix_inverse():
    input_matrix = [
        [1.0, 3.0, 5.3],
        [1.0, 2.89, 9.0],
        [3.2, 3.1, 2.9]
    ]
    inv_matrix = basics.get_matrix_inverse(input_matrix)

    expected_id_mat = basics.matmul(input_matrix, inv_matrix)

    print(f"Input matrix:\n{input_matrix}")
    print(f"Inverse matrix:\n{np.round(inv_matrix, 3)}")
    print(f"matmul result:\n{np.round(expected_id_mat, 3)}")


def demo_mlr_blue():
    # mock up inputs and outputs for multiple linear regression,
    # use BLUE formula with basics funcs to get coefficient estimates
    # note BLUE coefs B_hat given X, y = ((X_t * X) ** -1 ) * y

    # make ones for intercept
    x_ones = np.ones(10)
    x_0 = np.random.normal(0, 1, size=10)
    x_1 = np.random.exponential(size=10)
    x_2 = np.random.normal(2, 0.5, size=10)
    x_3 = np.random.binomial(1, 0.2, size=10)

    X = np.array([x_ones, x_0, x_1, x_2, x_3]).T

    # declare "true" target coefficients
    b_intercept = 42
    b_0 = 2
    b_1 = 1.4
    b_2 = 3.2
    b_3 = 0.4

    coefs = np.array([[b_intercept], [b_0], [b_1], [b_2], [b_3]])

    print(f"X shape: {X.shape}, coefs shape: {coefs.shape}")

    mock_y_observed_raw = basics.matmul(X, coefs)
    noise = np.random.normal(0, 0.2, size=10)
    mock_y_observed_w_noise = np.round(
        np.array(mock_y_observed_raw) + noise, 3
    )

    y = mock_y_observed_w_noise.copy()
    print(y)


    def get_shape(arr):
        print(np.array(arr).shape)

    # get OLS BLUE
    xT = basics.transpose(X)
    xTx = basics.matmul(xT, X)
    xTx_inv = basics.get_matrix_inverse(xTx)
    penultimate = basics.matmul(xTx_inv, xT)
#   still fighting dimension mismatch
#    blue = basics.matmul(penultimate, y)


if __name__ == "__main__":
    demo_matrix_inverse()
    demo_mlr_blue()

