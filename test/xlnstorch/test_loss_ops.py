import pytest
import torch
import xlnstorch as xltorch

def make_lns_tensor(data, f=23):
    return xltorch.lnstensor(data, f=f)

def verify_lns_loss(result, expected, msg=None, rtol=1e-5):
    assert isinstance(result, xltorch.LNSTensor), "Result should be an LNS tensor"
    assert torch.allclose(result.value, expected, rtol=rtol), msg or "Loss values don't match expected results"

@pytest.fixture
def sample_loss_tensors():
    x = make_lns_tensor([0.1, 0.5, 0.9], f=23)
    y = make_lns_tensor([0.2, 0.4, 0.8], f=23)
    return x, y

@pytest.fixture
def sample_classification_tensors():
    x = make_lns_tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], f=23)
    y = torch.tensor([0, 1], dtype=torch.long)
    return x, y

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_loss(sample_loss_tensors, reduction):
    x, y = sample_loss_tensors

    # Functional
    lns_result = torch.nn.functional.mse_loss(x, y, reduction=reduction)
    expected = torch.nn.functional.mse_loss(x.value, y.value, reduction=reduction)

    verify_lns_loss(lns_result, expected, f"MSE loss functional failed for reduction={reduction}")

    # Module
    mse = torch.nn.MSELoss(reduction=reduction)
    lns_result_mod = mse(x, y)
    expected_mod = mse(x.value, y.value)

    verify_lns_loss(lns_result_mod, expected_mod, f"MSE loss module failed for reduction={reduction}")

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_l1_loss(sample_loss_tensors, reduction):
    x, y = sample_loss_tensors

    # Functional
    lns_result = torch.nn.functional.l1_loss(x, y, reduction=reduction)
    expected = torch.nn.functional.l1_loss(x.value, y.value, reduction=reduction)

    verify_lns_loss(lns_result, expected, f"L1 loss functional failed for reduction={reduction}")

    # Module
    l1 = torch.nn.L1Loss(reduction=reduction)
    lns_result_mod = l1(x, y)
    expected_mod = l1(x.value, y.value)

    verify_lns_loss(lns_result_mod, expected_mod, f"L1 loss module failed for reduction={reduction}")

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_binary_cross_entropy(sample_loss_tensors, reduction):
    x, y = sample_loss_tensors

    # Clamp to avoid log(0)
    x_val = x.value.clamp(min=1e-6, max=1-1e-6)
    y_val = y.value.clamp(min=1e-6, max=1-1e-6)

    x = make_lns_tensor(x_val)
    y = make_lns_tensor(y_val)

    # Functional
    lns_result = torch.nn.functional.binary_cross_entropy(x, y, reduction=reduction)
    expected = torch.nn.functional.binary_cross_entropy(x.value, y.value, reduction=reduction)

    verify_lns_loss(lns_result, expected, f"BCE loss functional failed for reduction={reduction}")

    # Module
    bce = torch.nn.BCELoss(reduction=reduction)
    lns_result_mod = bce(x, y)
    expected_mod = bce(x.value, y.value)

    verify_lns_loss(lns_result_mod, expected_mod, f"BCE loss module failed for reduction={reduction}")

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_nll_loss(sample_classification_tensors, reduction):
    x, y = sample_classification_tensors

    # Functional
    lns_result = torch.nn.functional.nll_loss(x, y, reduction=reduction)
    expected = torch.nn.functional.nll_loss(x.value, y, reduction=reduction)

    verify_lns_loss(lns_result, expected, f"NLL loss functional failed for reduction={reduction}")

    # Module
    nll = torch.nn.NLLLoss(reduction=reduction)
    lns_result_mod = nll(x, y)
    expected_mod = nll(x.value, y)

    verify_lns_loss(lns_result_mod, expected_mod, f"NLL loss module failed for reduction={reduction}")
