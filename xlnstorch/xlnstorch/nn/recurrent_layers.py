import torch
from . import LNSModule
from xlnstorch import rand, zeros

class LNSRNN(LNSModule):

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
                weight_ih = rand(hidden_size, layer_input_size)
                weight_hh = rand(hidden_size, hidden_size)
                self.register_parameter(f"weight_ih_{suffix}", (weight_ih * 2 - 1) * sqrt_k)
                self.register_parameter(f"weight_hh_{suffix}", (weight_hh * 2 - 1) * sqrt_k)

                if bias:
                    bias_ih = rand(hidden_size)
                    bias_hh = rand(hidden_size)
                    self.register_parameter(f"bias_ih_{suffix}", (bias_ih * 2 - 1) * sqrt_k)
                    self.register_parameter(f"bias_hh_{suffix}", (bias_hh * 2 - 1) * sqrt_k)

        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, h0=None):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.shape

        if h0 is None:
            h0 = zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
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

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            nonlinearity: str = 'tanh',
        ):
        super().__init__()

        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError("Nonlinearity must be either 'tanh' or 'relu'")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        sqrt_k = 1.0 / (self.hidden_size ** 0.5)
        weight_ih = rand(hidden_size, input_size)
        weight_hh = rand(hidden_size, hidden_size)

        self.register_parameter("weight_ih", (weight_ih * 2 - 1) * sqrt_k)
        self.register_parameter("weight_hh", (weight_hh * 2 - 1) * sqrt_k)

        if bias:
            bias_ih = rand(hidden_size)
            bias_hh = rand(hidden_size)
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