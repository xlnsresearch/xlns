import pytest
import torch
import xlnstorch as xltorch

def make_lns_tensor(data, f=23):
    return xltorch.lnstensor(data, f=f)

def verify_lns_result(result, expected, msg=None, rtol=1e-5):
    assert isinstance(result, xltorch.LNSTensor), "Result should be an LNS tensor"
    assert torch.allclose(result.value, expected, rtol=rtol), msg or "Values don't match expected results"

@pytest.fixture
def sample_activation_tensor():
    return make_lns_tensor([-2.0, -0.5, 0.0, 0.5, 2.0], f=23)

@pytest.mark.parametrize("inplace", [False, True])
def test_relu(sample_activation_tensor, inplace):
    lns = sample_activation_tensor
    lns_in = lns.clone() if inplace else lns

    # Functional
    lns_result = torch.nn.functional.relu(lns_in, inplace=inplace)
    expected = torch.nn.functional.relu(lns.value, inplace=False)

    verify_lns_result(lns_result, expected, f"ReLU functional failed (inplace={inplace})")

    # Module
    relu_mod = torch.nn.ReLU(inplace=inplace)
    lns_mod = lns.clone() if inplace else lns
    lns_result_mod = relu_mod(lns_mod)
    expected_mod = relu_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"ReLU module failed (inplace={inplace})")

@pytest.mark.parametrize("negative_slope", [0.01, 0.2])
@pytest.mark.parametrize("inplace", [False, True])
def test_leaky_relu(sample_activation_tensor, negative_slope, inplace):
    lns = sample_activation_tensor
    lns_in = lns.clone() if inplace else lns

    # Functional
    lns_result = torch.nn.functional.leaky_relu(lns_in, negative_slope=negative_slope, inplace=inplace)
    expected = torch.nn.functional.leaky_relu(lns.value, negative_slope=negative_slope, inplace=False)

    verify_lns_result(lns_result, expected, f"LeakyReLU functional failed (inplace={inplace}, slope={negative_slope})")

    # Module
    leaky_mod = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    lns_mod = lns.clone() if inplace else lns
    lns_result_mod = leaky_mod(lns_mod)
    expected_mod = leaky_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"LeakyReLU module failed (inplace={inplace}, slope={negative_slope})")

@pytest.mark.parametrize("inplace", [False, True])
def test_threshold(sample_activation_tensor, inplace):
    lns = sample_activation_tensor
    threshold_val = 0.0
    value = -1.0
    lns_in = lns.clone() if inplace else lns

    # Functional
    lns_result = torch.nn.functional.threshold(lns_in, threshold=threshold_val, value=value, inplace=inplace)
    expected = torch.nn.functional.threshold(lns.value, threshold=threshold_val, value=value, inplace=False)

    verify_lns_result(lns_result, expected, f"Threshold functional failed (inplace={inplace})")

    # Module
    thresh_mod = torch.nn.Threshold(threshold=threshold_val, value=value, inplace=inplace)
    lns_mod = lns.clone() if inplace else lns
    lns_result_mod = thresh_mod(lns_mod)
    expected_mod = thresh_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"Threshold module failed (inplace={inplace})")

def test_tanh(sample_activation_tensor):
    lns = sample_activation_tensor

    # Functional
    lns_result = torch.nn.functional.tanh(lns)
    expected = torch.nn.functional.tanh(lns.value)

    verify_lns_result(lns_result, expected, "Tanh functional failed")

    # Module
    tanh_mod = torch.nn.Tanh()
    lns_result_mod = tanh_mod(lns)
    expected_mod = tanh_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, "Tanh module failed")

def test_sigmoid(sample_activation_tensor):
    lns = sample_activation_tensor

    # Functional
    lns_result = torch.nn.functional.sigmoid(lns)
    expected = torch.nn.functional.sigmoid(lns.value)

    verify_lns_result(lns_result, expected, "Sigmoid functional failed")

    # Module
    sig_mod = torch.nn.Sigmoid()
    lns_result_mod = sig_mod(lns)
    expected_mod = sig_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, "Sigmoid module failed")

def test_logsigmoid(sample_activation_tensor):
    lns = sample_activation_tensor

    # Functional
    lns_result = torch.nn.functional.logsigmoid(lns)
    expected = torch.nn.functional.logsigmoid(lns.value)

    verify_lns_result(lns_result, expected, "LogSigmoid functional failed")

    # Module
    logsig_mod = torch.nn.LogSigmoid()
    lns_result_mod = logsig_mod(lns)
    expected_mod = logsig_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, "LogSigmoid module failed")

@pytest.mark.parametrize("dim", [0, -1])
def test_softmin(dim):
    lns = make_lns_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

    # Functional
    lns_result = torch.nn.functional.softmin(lns, dim=dim)
    expected = torch.nn.functional.softmin(lns.value, dim=dim)

    verify_lns_result(lns_result, expected, f"Softmin functional failed (dim={dim})")

    # Module
    softmin_mod = torch.nn.Softmin(dim=dim)
    lns_result_mod = softmin_mod(lns)
    expected_mod = softmin_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"Softmin module failed (dim={dim})")

@pytest.mark.parametrize("dim", [0, -1])
def test_softmax(dim):
    lns = make_lns_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

    # Functional
    lns_result = torch.nn.functional.softmax(lns, dim=dim)
    expected = torch.nn.functional.softmax(lns.value, dim=dim)

    verify_lns_result(lns_result, expected, f"Softmax functional failed (dim={dim})")

    # Module
    softmax_mod = torch.nn.Softmax(dim=dim)
    lns_result_mod = softmax_mod(lns)
    expected_mod = softmax_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"Softmax module failed (dim={dim})")

@pytest.mark.parametrize("dim", [0, -1])
def test_log_softmax(dim):
    lns = make_lns_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

    # Functional
    lns_result = torch.nn.functional.log_softmax(lns, dim=dim)
    expected = torch.nn.functional.log_softmax(lns.value, dim=dim)

    verify_lns_result(lns_result, expected, f"LogSoftmax functional failed (dim={dim})")

    # Module
    logsoftmax_mod = torch.nn.LogSoftmax(dim=dim)
    lns_result_mod = logsoftmax_mod(lns)
    expected_mod = logsoftmax_mod(lns.value)

    verify_lns_result(lns_result_mod, expected_mod, f"LogSoftmax module failed (dim={dim})")