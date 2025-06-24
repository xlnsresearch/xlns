import torch
import xlnstorch as xltorch

input = xltorch.randn(3, requires_grad=True)
target = xltorch.lnstensor([1.0, 1.0, 1.0])

model = xltorch.layers.LNSLinear(3, 3, bias=True)
# here we use the custom LNS optimizer method `model.parameter_groups()`
optimizer = xltorch.optimizers.LNSSGD(model.parameter_groups(), lr=0.1)
loss_fn = torch.nn.MSELoss(reduction='mean') # we can use standard PyTorch loss classes

for i in range(20):
    optimizer.zero_grad()

    output = model(input)
    loss = loss_fn(output, target)

    print(f"Iteration {i + 1}, Loss: {loss.item():.2E}")

    loss.backward()
    optimizer.step()