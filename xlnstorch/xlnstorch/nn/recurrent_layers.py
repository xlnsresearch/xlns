from typing import Optional, Union
import torch
from . import LNSModule
from xlnstorch import rand, zeros

class LNSRNN(LNSModule):
    r"""
    An LNS multi-layer Elman RNN.

    See also: :py:class:`torch.nn.RNN`

    Parameters
    ----------
    input_size : int
        The number of expected features in the input tensor.
    hidden_size : int
        The number of features in the hidden state.
    num_layers : int, optional
        The number of recurrent layers to stack. Default: 1.
    nonlinearity : str, optional
        The nonlinearity to use. Either 'tanh' or 'relu'. Default: 'tanh'.
    bias : bool, optional
        If True, adds a learnable bias to the layer. Default: True.
    batch_first : bool, optional
        If True, the input and output tensors are provided as (batch, seq, feature).
        If False, they are provided as (seq, batch, feature). Default: False.
    dropout : float, optional
        If non-zero, introduces a dropout layer on the outputs of each RNN layer except the
        last layer. Default: 0.0.
    bidirectional : bool, optional
        If True, becomes a bidirectional RNN. Default: False.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih_l{k} : LNSTensor
        The input-hidden weights of the kth layer with shape
        :math:`(\text{hidden_size}, \text{num_directions} \cdot \text{hidden_size})`.
    weight_hh_l{k} : LNSTensor
        The hidden-hidden weights of the kth layer with shape
        :math:`(\text{hidden_size}, \text{hidden_size})`.
    bias_ih_l{k} : LNSTensor, optional
        The input-hidden bias of the kth layer with shape
        :math:`(\text{num_directions} \cdot \text{hidden_size})`.
    bias_hh_l{k} : LNSTensor, optional
        The hidden-hidden bias of the kth layer with shape
        :math:`(\text{hidden_size})`.

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.

    Bidirectional RNNs have two sets of weights and biases for each layer.
    The backward direction weights and biases are suffixed with `_reverse`
    and have the same shapes as the forward direction weights and biases.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            nonlinearity: str = 'tanh',
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
    ):
        super().__init__()

        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError("Nonlinearity must be either 'tanh' or 'relu'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

                sqrt_k = 1.0 / (self.hidden_size ** 0.5)
                weight_ih = rand(hidden_size, layer_input_size, f=weight_f, b=weight_b)
                weight_hh = rand(hidden_size, hidden_size, f=weight_f, b=weight_b)
                self.register_parameter(f"weight_ih_{suffix}", (weight_ih * 2 - 1) * sqrt_k)
                self.register_parameter(f"weight_hh_{suffix}", (weight_hh * 2 - 1) * sqrt_k)

                if bias:
                    bias_ih = rand(hidden_size, f=bias_f, b=bias_b)
                    bias_hh = rand(hidden_size, f=bias_f, b=bias_b)
                    self.register_parameter(f"bias_ih_{suffix}", (bias_ih * 2 - 1) * sqrt_k)
                    self.register_parameter(f"bias_hh_{suffix}", (bias_hh * 2 - 1) * sqrt_k)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, h0=None):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if h0 is None:
            h0 = zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, b=x.base)
        else:
            assert h0.shape == (self.num_layers * self.num_directions, batch_size, self.hidden_size)

        h_n = []
        layer_input = x

        for layer in range(self.num_layers):
            layer_outputs = []

            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                w_ih = getattr(self, f"weight_ih_{suffix}")
                w_hh = getattr(self, f"weight_hh_{suffix}")
                b_ih = getattr(self, f"bias_ih_{suffix}") if self.bias else None
                b_hh = getattr(self, f"bias_hh_{suffix}") if self.bias else None

                h_t = h0[layer * self.num_directions + direction]

                time_iter = range(seq_len)
                if direction == 1:
                    time_iter = reversed(time_iter)

                outputs = []
                for t in time_iter:
                    inp = layer_input[t]

                    result_ih = torch.nn.functional.linear(inp, w_ih, b_ih)
                    result_hh = torch.nn.functional.linear(h_t, w_hh, b_hh)
                    h_t = result_ih + result_hh

                    if self.nonlinearity == 'tanh':
                        h_t = torch.nn.functional.tanh(h_t)
                    elif self.nonlinearity == 'relu':
                        h_t = torch.nn.functional.relu(h_t)

                    outputs.append(h_t)

                if direction == 1:
                    outputs = outputs[::-1]

                layer_outputs.append(torch.stack(outputs, 0))
                h_n.append(h_t)

            layer_output = (
                torch.cat(layer_outputs, dim=2)
                if self.num_directions == 2
                else layer_outputs[0]
            )

            if layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)

            layer_input = layer_output

        output = layer_input
        h_n = torch.stack(h_n, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

class LNSRNNCell(LNSModule):
    r"""
    An LNS Elman RNN cell.

    See also: :py:class:`torch.nn.RNNCell`

    Parameters
    ----------
    input_size : int
        Number of expected features in the input tensor $x_t$.
    hidden_size : int
        Number of features in the hidden state $h_t$.
    bias : bool, optional
        If ``True``, adds learnable bias terms. Default: ``True``.
    nonlinearity : str, optional
        Non-linear activation to apply. Either ``'tanh'`` or ``'relu'``.
        Default: ``'tanh'``.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih : LNSTensor
        The input-hidden weights with shape
        :math:`(\text{hidden_size}, \text{input_size})`.
    weight_hh : LNSTensor
        The hidden-hidden weights with shape
        :math:`(\text{hidden_size}, \text{hidden_size})`.
    bias_ih : LNSTensor, optional
        The input-hidden bias with shape
        :math:`(\text{hidden_size})`.
    bias_hh : LNSTensor, optional
        The hidden-hidden bias with shape
        :math:`(\text{hidden_size})`.

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            nonlinearity: str = 'tanh',
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
        ):
        super().__init__()

        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError("Nonlinearity must be either 'tanh' or 'relu'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        sqrt_k = 1.0 / (self.hidden_size ** 0.5)
        weight_ih = rand(hidden_size, input_size, f=weight_f, b=weight_b)
        weight_hh = rand(hidden_size, hidden_size, f=weight_f, b=weight_b)

        self.register_parameter("weight_ih", (weight_ih * 2 - 1) * sqrt_k)
        self.register_parameter("weight_hh", (weight_hh * 2 - 1) * sqrt_k)

        if bias:
            bias_ih = rand(hidden_size, f=bias_f, b=bias_b)
            bias_hh = rand(hidden_size, f=bias_f, b=bias_b)
            self.register_parameter("bias_ih", (bias_ih * 2 - 1) * sqrt_k)
            self.register_parameter("bias_hh", (bias_hh * 2 - 1) * sqrt_k)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):

        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)

            if hx is not None:
                hx = hx.unsqueeze(0)

        h0 = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
        h1 = torch.nn.functional.linear(hx, self.weight_hh, self.bias_hh) if hx is not None else 0
        h = h0 + h1

        if self.nonlinearity == 'tanh':
            h = torch.nn.functional.tanh(h)
        elif self.nonlinearity == 'relu':
            h = torch.nn.functional.relu(h)

        if not is_batched:
            h = h.squeeze(0)

        return h

class LNSLSTM(LNSModule):
    r"""
    An LNS multi-layer long-short term memory (LSTM) RNN.

    See also: :py:class:`torch.nn.LSTM`

    Parameters
    ----------
    input_size : int
        The number of expected features in the input tensor.
    hidden_size : int
        The number of features in the hidden state.
    num_layers : int, optional
        The number of recurrent layers to stack. Default: 1.
    bias : bool, optional
        If True, adds a learnable bias to the layer. Default: True.
    batch_first : bool, optional
        If True, the input and output tensors are provided as (batch, seq, feature).
        If False, they are provided as (seq, batch, feature). Default: False.
    dropout : float, optional
        If non-zero, introduces a dropout layer on the outputs of each LSTM layer except the
        last layer. Default: 0.0.
    bidirectional : bool, optional
        If True, becomes a bidirectional LSTM. Default: False.
    proj_size : int, optional
        If > 0, use LSTM with projections. The hidden state h_t will have size
        ``proj_size`` instead of ``hidden_size``. Default: 0.
        Note: must satisfy 0 <= proj_size < hidden_size.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih_l{k} : LNSTensor
        The input-hidden weights of the kth layer with shape
        :math:`(4 \cdot \text{hidden_size}, \text{layer_input_size})`,
        where layer_input_size is input_size for layer 0, else
        :math:`\text{num_directions} \cdot (\text{proj_size}
        \textit{ if} \text{proj_size} > 0 \textit{else} \text{hidden_size})`.
    weight_hh_l{k} : LNSTensor
        The hidden-hidden weights of the kth layer with shape
        :math:`(4 \cdot \text{hidden_size}, (\text{proj_size}
        \textit{ if} \text{proj_size} > 0 \textit{else} \text{hidden_size})`.
    bias_ih_l{k} : LNSTensor, optional
        The input-hidden bias of the kth layer with shape
        :math:`(4 \cdot (\text{proj_size} \textit{ if}
        \text{proj_size} > 0 \textit{else} \text{hidden_size})`.
    bias_hh_l{k} : LNSTensor, optional
        The hidden-hidden bias of the kth layer with shape
        :math:`(4 \cdot \text{hidden_size})`.
    weight_hr_l{k} : LNSTensor, optional
        The projection matrix of the kth layer with shape (proj_size, hidden_size).
        This is present only if :math:`\text{proj_size} > 0`

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.

    Bidirectional LSTMs have two sets of weights and biases for each layer.
    The backward direction weights and biases are suffixed with `_reverse`
    and have the same shapes as the forward direction weights and biases.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            proj_size: int = 0,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.hidden_out_size = self.proj_size if self.proj_size > 0 else self.hidden_size

        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                layer_input_size = input_size if layer == 0 else self.hidden_out_size * self.num_directions

                sqrt_k = 1.0 / (self.hidden_size ** 0.5)
                weight_ih = rand(4 * hidden_size, layer_input_size, f=weight_f, b=weight_b)
                weight_hh = rand(4 * hidden_size, self.hidden_out_size, f=weight_f, b=weight_b)
                self.register_parameter(f"weight_ih_{suffix}", (weight_ih * 2 - 1) * sqrt_k)
                self.register_parameter(f"weight_hh_{suffix}", (weight_hh * 2 - 1) * sqrt_k)

                if self.proj_size > 0:
                    weight_hr = rand(self.proj_size, self.hidden_size, f=weight_f, b=weight_b)
                    self.register_parameter(f"weight_hr_{suffix}", (weight_hr * 2 - 1) * sqrt_k)

                if bias:
                    bias_ih = rand(4 * hidden_size, f=bias_f, b=bias_b)
                    bias_hh = rand(4 * hidden_size, f=bias_f, b=bias_b)
                    self.register_parameter(f"bias_ih_{suffix}", (bias_ih * 2 - 1) * sqrt_k)
                    self.register_parameter(f"bias_hh_{suffix}", (bias_hh * 2 - 1) * sqrt_k)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, hx=None):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h0 = zeros(self.num_layers * self.num_directions, batch_size, self.hidden_out_size, b=x.base)
            c0 = zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, b=x.base)
        else:
            if not (isinstance(hx, tuple) and len(hx) == 2):
                raise ValueError("hx must be a tuple (h0, c0)")
            h0, c0 = hx
            assert h0.shape == (self.num_layers * self.num_directions, batch_size, self.hidden_out_size)
            assert c0.shape == (self.num_layers * self.num_directions, batch_size, self.hidden_size)

        h_n = []
        c_n = []
        layer_input = x

        for layer in range(self.num_layers):
            layer_outputs = []

            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                w_ih = getattr(self, f"weight_ih_{suffix}")
                w_hh = getattr(self, f"weight_hh_{suffix}")
                w_hr = getattr(self, f"weight_hr_{suffix}", None)
                b_ih = getattr(self, f"bias_ih_{suffix}") if self.bias else None
                b_hh = getattr(self, f"bias_hh_{suffix}") if self.bias else None

                h_t = h0[layer * self.num_directions + direction]
                c_t = c0[layer * self.num_directions + direction]

                time_iter = range(seq_len)
                if direction == 1:
                    time_iter = reversed(time_iter)

                outputs = []
                for t in time_iter:
                    inp = layer_input[t]

                    gates = torch.nn.functional.linear(inp, w_ih, b_ih)
                    gates += torch.nn.functional.linear(h_t, w_hh, b_hh)

                    i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

                    i_t = torch.sigmoid(i_gate)
                    f_t = torch.sigmoid(f_gate)
                    g_t = torch.tanh(g_gate)
                    o_t = torch.sigmoid(o_gate)

                    c_t = f_t * c_t + i_t * g_t
                    h_hat = o_t * torch.tanh(c_t)

                    if self.proj_size > 0:
                        h_t = torch.nn.functional.linear(h_hat, w_hr, None)
                    else:
                        h_t = h_hat

                    outputs.append(h_t)

                if direction == 1:
                    outputs = outputs[::-1]

                layer_outputs.append(torch.stack(outputs, 0))
                h_n.append(h_t)
                c_n.append(c_t)

            layer_output = (
                torch.cat(layer_outputs, dim=2)
                if self.num_directions == 2
                else layer_outputs[0]
            )

            if layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)

            layer_input = layer_output

        output = layer_input
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)

class LNSLSTMCell(LNSModule):
    r"""
    An LNS long-short term memory (LSTM) cell.

    See also: :py:class:`torch.nn.LSTMCell`

    Parameters
    ----------
    input_size : int
        Number of expected features in the input tensor $x_t$.
    hidden_size : int
        Number of features in the hidden state $h_t$.
    bias : bool, optional
        If ``True``, adds learnable bias terms. Default: ``True``.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih : LNSTensor
        The input-hidden weights with shape
        :math:`(4 \cdot \text{hidden_size}, \text{input_size})`.
    weight_hh : LNSTensor
        The hidden-hidden weights with shape
        :math:`(4 \cdot \text{hidden_size}, \text{hidden_size})`.
    bias_ih : LNSTensor, optional
        The input-hidden bias with shape
        :math:`(4 \cdot \text{hidden_size})`.
    bias_hh : LNSTensor, optional
        The hidden-hidden bias with shape
        :math:`(4 \cdot \text{hidden_size})`.

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        sqrt_k = 1.0 / (self.hidden_size ** 0.5)

        weight_ih = rand(4 * hidden_size, input_size, f=weight_f, b=weight_b)
        weight_hh = rand(4 * hidden_size, hidden_size, f=weight_f, b=weight_b)
        self.register_parameter("weight_ih", (weight_ih * 2 - 1) * sqrt_k)
        self.register_parameter("weight_hh", (weight_hh * 2 - 1) * sqrt_k)

        if bias:
            bias_ih = rand(4 * hidden_size, f=bias_f, b=bias_b)
            bias_hh = rand(4 * hidden_size, f=bias_f, b=bias_b)
            self.register_parameter("bias_ih", (bias_ih * 2 - 1) * sqrt_k)
            self.register_parameter("bias_hh", (bias_hh * 2 - 1) * sqrt_k)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        if hx is not None and (not isinstance(hx, tuple) or len(hx) != 2):
                raise ValueError("hx must be a tuple of (h_0, c_0) for LSTMCell")

        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)

            if hx is not None:
                hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0))

        h_prev, c_prev = (None, None) if hx is None else hx
        if c_prev is None:
            c_prev = zeros(x.size(0), self.hidden_size, b=x.base)

        gates = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
        if h_prev is not None:
            gates = gates + torch.nn.functional.linear(h_prev, self.weight_hh, self.bias_hh)

        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=-1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        if not is_batched:
            h_t = h_t.squeeze(0)
            c_t = c_t.squeeze(0)

        return h_t, c_t

class LNSGRU(LNSModule):
    r"""
    An LNS multi-layer gated recurrent unit (GRU) RNN.

    See also: :py:class:`torch.nn.GRU`

    Parameters
    ----------
    input_size : int
        The number of expected features in the input tensor.
    hidden_size : int
        The number of features in the hidden state.
    num_layers : int, optional
        The number of recurrent layers to stack. Default: 1.
    bias : bool, optional
        If True, adds a learnable bias to the layer. Default: True.
    batch_first : bool, optional
        If True, the input and output tensors are provided as (batch, seq, feature).
        If False, they are provided as (seq, batch, feature). Default: False.
    dropout : float, optional
        If non-zero, introduces a dropout layer on the outputs of each GRU layer except the
        last layer. Default: 0.0.
    bidirectional : bool, optional
        If True, becomes a bidirectional GRU. Default: False.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih_l{k} : LNSTensor
        The input-hidden weights of the kth layer with shape
        :math:`(3 \cdot \text{hidden_size}, \text{layer_input_size})`,
        where layer_input_size is input_size for layer 0, else
        :math:`\text{num_directions} \cdot \text{hidden_size}`.
    weight_hh_l{k} : LNSTensor
        The hidden-hidden weights of the kth layer with shape
        :math:`(3 \cdot \text{hidden_size}, \text{hidden_size})`.
    bias_ih_l{k} : LNSTensor, optional
        The input-hidden bias of the kth layer with shape
        :math:`(3 \cdot \text{hidden_size})`.
    bias_hh_l{k} : LNSTensor, optional
        The hidden-hidden bias of the kth layer with shape
        :math:`(3 \cdot \text{hidden_size})`.

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.

    Bidirectional GRUs have two sets of weights and biases for each layer.
    The backward direction weights and biases are suffixed with `_reverse`
    and have the same shapes as the forward direction weights and biases.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            dropout: float = 0.0,
            bidirectional: bool = False,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

                sqrt_k = 1.0 / (self.hidden_size ** 0.5)
                weight_ih = rand(3 * hidden_size, layer_input_size, f=weight_f, b=weight_b)
                weight_hh = rand(3 * hidden_size, hidden_size, f=weight_f, b=weight_b)
                self.register_parameter(f"weight_ih_{suffix}", (weight_ih * 2 - 1) * sqrt_k)
                self.register_parameter(f"weight_hh_{suffix}", (weight_hh * 2 - 1) * sqrt_k)

                if bias:
                    bias_ih = rand(3 * hidden_size, f=bias_f, b=bias_b)
                    bias_hh = rand(3 * hidden_size, f=bias_f, b=bias_b)
                    self.register_parameter(f"bias_ih_{suffix}", (bias_ih * 2 - 1) * sqrt_k)
                    self.register_parameter(f"bias_hh_{suffix}", (bias_hh * 2 - 1) * sqrt_k)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if hx is None:
            h0 = zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, b=x.base)
        else:
            if isinstance(hx, tuple):
                raise ValueError("hx must be a single tensor (h0) for GRU")
            h0 = hx
            assert h0.shape == (self.num_layers * self.num_directions, batch_size, self.hidden_size)

        h_n = []
        layer_input = x

        for layer in range(self.num_layers):
            layer_outputs = []

            for direction in range(self.num_directions):
                suffix = f"l{layer}" + ("_reverse" if direction == 1 else "")
                w_ih = getattr(self, f"weight_ih_{suffix}")
                w_hh = getattr(self, f"weight_hh_{suffix}")
                b_ih = getattr(self, f"bias_ih_{suffix}") if self.bias else None
                b_hh = getattr(self, f"bias_hh_{suffix}") if self.bias else None

                h_t = h0[layer * self.num_directions + direction]

                time_iter = range(seq_len)
                if direction == 1:
                    time_iter = reversed(time_iter)

                outputs = []
                for t in time_iter:
                    inp = layer_input[t]

                    gi = torch.nn.functional.linear(inp, w_ih, b_ih)
                    gh = torch.nn.functional.linear(h_t, w_hh, b_hh)

                    i_r, i_z, i_n = gi.chunk(3, dim=-1)
                    h_r, h_z, h_n_part = gh.chunk(3, dim=-1)

                    r_t = torch.sigmoid(i_r + h_r)
                    z_t = torch.sigmoid(i_z + h_z)
                    n_t = torch.tanh(i_n + r_t * h_n_part)

                    h_t = (1.0 - z_t) * n_t + z_t * h_t

                    outputs.append(h_t)

                if direction == 1:
                    outputs = outputs[::-1]

                layer_outputs.append(torch.stack(outputs, 0))
                h_n.append(h_t)

            layer_output = (
                torch.cat(layer_outputs, dim=2)
                if self.num_directions == 2
                else layer_outputs[0]
            )

            if layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)

            layer_input = layer_output

        output = layer_input
        h_n = torch.stack(h_n, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

class LNSGRUCell(LNSModule):
    r"""
    An LNS gated recurrent unit (GRU) cell.

    See also: :py:class:`torch.nn.GRUCell`

    Parameters
    ----------
    input_size : int
        Number of expected features in the input tensor $x_t$.
    hidden_size : int
        Number of features in the hidden state $h_t$.
    bias : bool, optional
        If True, adds learnable bias terms. Default: True.
    weight_f : int, optional
        The number of fractional exponent bits for the weights. mutually exclusive with ``weight_b``.
    weight_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the weights; mutually exclusive with ``weight_f``.
    bias_f : int, optional
        The number of fractional exponent bits for the biases. mutually exclusive with ``bias_b``.
    bias_b : Optional[Union[float, int, torch.Tensor]], optional
        The explicit logarithm base for the biases; mutually exclusive with ``bias_f``.

    Attributes
    ----------
    weight_ih : LNSTensor
        The input-hidden weights with shape
        :math:`(3 \cdot \text{hidden_size}, \text{input_size})`.
    weight_hh : LNSTensor
        The hidden-hidden weights with shape
        :math:`(3 \cdot \text{hidden_size}, \text{input_size})`.
    bias_ih : LNSTensor, optional
        The input-hidden bias with shape
        :math:`(3 \cdot \text{hidden_size}, \text{input_size})`.
    bias_hh : LNSTensor, optional
        The hidden-hidden bias with shape
        :math:`(3 \cdot \text{hidden_size}, \text{input_size})`.

    Notes
    -----
    The weights and biases are initialized with random values uniformly
    distributed between :math:`-\sqrt{k}` and :math:`\sqrt{k}`, where
    :math:`k = \frac{1}{\text{hidden_size}}`.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            weight_f: int = None,
            weight_b: float = None,
            bias_f: int = None,
            bias_b: float = None,
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        sqrt_k = 1.0 / (self.hidden_size ** 0.5)

        weight_ih = rand(3 * hidden_size, input_size, f=weight_f, b=weight_b)
        weight_hh = rand(3 * hidden_size, hidden_size, f=weight_f, b=weight_b)
        self.register_parameter("weight_ih", (weight_ih * 2 - 1) * sqrt_k)
        self.register_parameter("weight_hh", (weight_hh * 2 - 1) * sqrt_k)

        if bias:
            bias_ih = rand(3 * hidden_size, f=bias_f, b=bias_b)
            bias_hh = rand(3 * hidden_size, f=bias_f, b=bias_b)
            self.register_parameter("bias_ih", (bias_ih * 2 - 1) * sqrt_k)
            self.register_parameter("bias_hh", (bias_hh * 2 - 1) * sqrt_k)

        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, hx=None):
        if hx is not None and isinstance(hx, tuple):
            raise ValueError("hx must be a single tensor h_0 for GRUCell")

        is_batched = x.dim() == 2
        if not is_batched:
            x = x.unsqueeze(0)
            if hx is not None:
                hx = hx.unsqueeze(0)

        if hx is None:
            h_prev = zeros(x.size(0), self.hidden_size, b=x.base)
        else:
            h_prev = hx

        gi = torch.nn.functional.linear(x, self.weight_ih, self.bias_ih)
        gh = torch.nn.functional.linear(h_prev, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)

        r_t = torch.sigmoid(i_r + h_r)
        z_t = torch.sigmoid(i_z + h_z)
        n_t = torch.tanh(i_n + r_t * h_n)

        h_t = (1.0 - z_t) * n_t + z_t * h_prev

        if not is_batched:
            h_t = h_t.squeeze(0)

        return h_t