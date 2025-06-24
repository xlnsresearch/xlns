import pytest
import torch
import xlnstorch as xltorch

def verify_lns_result(result, expected, msg=None):
    assert isinstance(result, xltorch.LNSTensor), "Result should be an LNS tensor"
    assert torch.allclose(result.value, expected, rtol=1e-5), msg or "Values don't match expected results"

class TestLNSLinear:
    def test_basic_linear(self):
        x = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        weight = xltorch.lnstensor([[5.0, 6.0], [7.0, 8.0]], f=23)
        bias = xltorch.lnstensor([1.0, 2.0], f=23)

        result = torch.nn.functional.linear(x, weight, bias)
        expected = torch.nn.functional.linear(x.value, weight.value, bias.value)

        verify_lns_result(result, expected, "Basic linear failed")
        assert result.shape == expected.shape

    def test_linear_no_bias(self):
        x = xltorch.lnstensor([[1.0, 2.0]], f=23)
        weight = xltorch.lnstensor([[1.0, 0.0], [0.0, 1.0]], f=23)

        result = torch.nn.functional.linear(x, weight)
        expected = torch.nn.functional.linear(x.value, weight.value)

        verify_lns_result(result, expected, "Linear no bias failed")
        assert result.shape == expected.shape

    def test_linear_broadcast(self):
        x = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        weight = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        bias = xltorch.lnstensor([1.0, 2.0], f=23)

        result = torch.nn.functional.linear(x, weight, bias)
        expected = torch.nn.functional.linear(x.value, weight.value, bias.value)

        verify_lns_result(result, expected, "Linear broadcast failed")

    def test_linear_module(self):
        x = xltorch.lnstensor([[1.0, 2.0], [3.0, 4.0]], f=23)
        mod = torch.nn.Linear(2, 2)

        # Set weights and bias to known values for reproducibility
        with torch.no_grad():
            mod.weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
            mod.bias.copy_(torch.tensor([1.0, 2.0]))

        # Use LNS tensor for input
        result = mod(x)
        expected = mod(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        verify_lns_result(result, expected, "Linear module failed")
        assert result.shape == expected.shape

class TestLNSBilinear:
    def test_basic_bilinear(self):
        x = xltorch.lnstensor([1.0, 2.0], f=23)
        y = xltorch.lnstensor([3.0, 4.0], f=23)
        weight = xltorch.lnstensor([[[1.0, 2.0], [3.0, 4.0]]], f=23)
        bias = xltorch.lnstensor([1.0], f=23)

        result = torch.nn.functional.bilinear(x, y, weight, bias)
        expected = torch.nn.functional.bilinear(x.value, y.value, weight.value, bias.value)

        verify_lns_result(result, expected, "Basic bilinear failed")

    def test_bilinear_no_bias(self):
        x = xltorch.lnstensor([1.0, 2.0], f=23)
        y = xltorch.lnstensor([3.0, 4.0], f=23)
        weight = xltorch.lnstensor([[[1.0, 2.0], [3.0, 4.0]]], f=23)

        result = torch.nn.functional.bilinear(x, y, weight)
        expected = torch.nn.functional.bilinear(x.value, y.value, weight.value)

        verify_lns_result(result, expected, "Bilinear no bias failed")

    def test_bilinear_module(self):
        x = xltorch.lnstensor([[1.0, 2.0]], f=23)
        y = xltorch.lnstensor([[3.0, 4.0]], f=23)

        mod = torch.nn.Bilinear(2, 2, 1)
        with torch.no_grad():
            mod.weight.copy_(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
            mod.bias.copy_(torch.tensor([1.0]))

        result = mod(x, y)
        expected = mod(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))

        verify_lns_result(result, expected, "Bilinear module failed")
        assert result.shape == expected.shape

class TestLNSDropout:
    def test_dropout_zero(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
        result = torch.nn.functional.dropout(x, p=0.0, training=True)

        verify_lns_result(result, x.value, "Dropout p=0 failed")

    def test_dropout_one(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
        result = torch.nn.functional.dropout(x, p=1.0, training=True)
        expected = torch.nn.functional.dropout(x.value, p=1.0, training=True)

        verify_lns_result(result, expected, "Dropout p=1 failed")

    def test_dropout_eval(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)
        result = torch.nn.functional.dropout(x, p=0.5, training=False)

        verify_lns_result(result, x.value, "Dropout eval mode failed")

    def test_dropout1d_shape(self):
        x = xltorch.lnstensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)
        result = torch.nn.functional.dropout1d(x, p=0.5, training=True)

        assert result.shape == x.shape

    def test_dropout2d_shape(self):
        x = xltorch.lnstensor(torch.ones(1, 2, 3, 4), f=23)
        result = torch.nn.functional.dropout2d(x, p=0.5, training=True)

        assert result.shape == x.shape

    def test_dropout3d_shape(self):
        x = xltorch.lnstensor(torch.ones(2, 3, 4, 5), f=23)
        result = torch.nn.functional.dropout3d(x, p=0.5, training=True)

        assert result.shape == x.shape

    def test_dropout_invalid_p(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)

        with pytest.raises(ValueError):
            torch.nn.functional.dropout(x, p=-0.1)

        with pytest.raises(ValueError):
            torch.nn.functional.dropout(x, p=1.1)

    def test_dropout1d_invalid_shape(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)

        with pytest.raises(ValueError):
            torch.nn.functional.dropout1d(x)

    def test_dropout2d_invalid_shape(self):
        x = xltorch.lnstensor([1.0, 2.0], f=23)

        with pytest.raises(ValueError):
            torch.nn.functional.dropout2d(x)

    def test_dropout3d_invalid_shape(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)

        with pytest.raises(ValueError):
            torch.nn.functional.dropout3d(x)

    def test_dropout_module(self):
        x = xltorch.lnstensor([1.0, 2.0, 3.0], f=23)

        mod = torch.nn.Dropout(p=0.5)
        mod.train()
        result = mod(x)

        # Can't check exact values due to randomness, but shape and type should match
        assert isinstance(result, xltorch.LNSTensor)
        assert result.shape == x.shape

    def test_dropout1d_module(self):
        x = xltorch.lnstensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], f=23)

        mod = torch.nn.Dropout1d(p=0.5)
        mod.train()
        result = mod(x)

        assert isinstance(result, xltorch.LNSTensor)
        assert result.shape == x.shape

    def test_dropout2d_module(self):
        x = xltorch.lnstensor(torch.ones(1, 2, 3, 4), f=23)

        mod = torch.nn.Dropout2d(p=0.5)
        mod.train()
        result = mod(x)

        assert isinstance(result, xltorch.LNSTensor)
        assert result.shape == x.shape

    def test_dropout3d_module(self):
        x = xltorch.lnstensor(torch.ones(2, 3, 4, 5), f=23)

        mod = torch.nn.Dropout3d(p=0.5)
        mod.train()
        result = mod(x)

        assert isinstance(result, xltorch.LNSTensor)
        assert result.shape == x.shape

# Add more tests for conv1d, etc. as needed following this style (once
# conv1d is fully fixed).