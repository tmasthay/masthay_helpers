from hypothesis import given, strategies as st
import torch
from masthay_helpers.typlotlib import plot_tensor2d_subplot


def dim_st(min_dims=4, max_dims=8, min_val=1, max_val=10):
    return st.integers(min_value=min_dims, max_value=max_dims).flatmap(
        lambda dims: st.lists(
            st.integers(min_value=min_val, max_value=max_val), min_size=dims
        )
    )


def tensor_st(min_dims=4, max_dims=8, min_val=1, max_val=10):
    return dim_st().map(lambda shape: torch.rand(*shape))


# @given(tensor_st(min_dims=3, max_dims=5, min_val=2, max_val=4))
def test_plot_tensor2d_subplot(tensor):
    input(tensor)
    plot_tensor2d_subplot(
        tensor=tensor, labels=[f'dim_{i}' for i in range(len(tensor.shape))]
    )


def main():
    tensor = torch.rand(6, 5, 5, 2, 2, 3)
    labels = [f'dim_{i}' for i in range(len(tensor.shape[1:]))]
    final_labels = [[f'SUBPLOT_NAME_{i}'] + labels for i in range(tensor.shape[0])]

    layout_args={'pad': 3.0, 'h_pad': 5.0, 'w_pad': 0.0}

    plot_tensor2d_subplot(tensor=tensor, labels=final_labels, print_freq=1, verbose=True, subplot_shape=(3,2), layout_args=layout_args, duration=400)


if __name__ == '__main__':
    main()
