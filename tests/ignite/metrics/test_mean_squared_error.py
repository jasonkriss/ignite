from ignite.exceptions import NotComputableError
from ignite.metrics import MeanSquaredError
import pytest
import torch


def test_zero_div():
    mse = MeanSquaredError()
    with pytest.raises(NotComputableError):
        mse.compute()


def test_compute():
    mse = MeanSquaredError()

    y_pred = torch.Tensor([[2.0], [-2.0]])
    y = torch.zeros(2)
    mse.update((y_pred, y))
    assert mse.compute() == 4.0

    mse.reset()
    y_pred = torch.Tensor([[3.0], [-3.0]])
    y = torch.zeros(2)
    mse.update((y_pred, y))
    assert mse.compute() == 9.0


def test_multioutput_compute():
    mse = MeanSquaredError()

    y_pred = torch.Tensor([[0, 2], [-1, 2], [8, -5]])
    y = torch.Tensor([[0.5, 1], [-1, 1], [7, -6]])
    mse.update((y_pred, y))
    assert mse.compute() == pytest.approx(0.708333)
