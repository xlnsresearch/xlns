import torch
import xlnstorch as xlt

class TestModel(xlt.nn.LNSModule):

    def __init__(self):
        super().__init__()
        self.fc1 = xlt.nn.LNSLinear(10, 5)
        self.tanh = torch.nn.Tanh()
        self.fc2 = xlt.nn.LNSLinear(5, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

input = xlt.full((10,), fill_value=0.5)
target = xlt.lnstensor([1.])

model = TestModel()
optimizer = xlt.optim.LNSSGD(model.lns_parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss(reduction='mean')

for _ in range(20):
    optimizer.zero_grad()

    output = model(input)
    loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Loss after training:", loss_fn(model(input), target).item())
torch.save(model.state_dict(), "model.pth")

model2 = TestModel()
model2.load_state_dict(torch.load("model.pth", weights_only=True))
model2.eval()

print("Loss after loading model:", loss_fn(model2(input), target).item())

# Uncomment the following lines to save/load the full model
# when full pickling is supported.
# torch.save(model, "model_full.pth")
# model3 = torch.load("model_full.pth", weights_only=False)
# model3.eval()

# print("Loss after loading full model:", loss_fn(model3(input), target).item())
