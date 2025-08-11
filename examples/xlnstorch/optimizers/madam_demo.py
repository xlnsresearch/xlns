import torch
import matplotlib.pyplot as plt
import xlnstorch as xltorch

torch.manual_seed(0)

inp = xltorch.randn(10)
target = xltorch.ones(5)

f = 8

model_sgd = xltorch.nn.LNSLinear(10, 5, bias=True, weight_f=f, bias_f=f)
model_mul = xltorch.nn.LNSLinear(10, 5, bias=True, weight_f=f, bias_f=f)
model_mul_pow = xltorch.nn.LNSLinear(10, 5, bias=True, weight_f=f, bias_f=f)

model_mul.weight_lns.data.copy_(model_sgd.weight_lns.data)
model_mul.bias_lns.data.copy_(model_sgd.bias_lns.data)
model_mul_pow.weight_lns.data.copy_(model_sgd.weight_lns.data)
model_mul_pow.bias_lns.data.copy_(model_sgd.bias_lns.data)

opt_sgd = xltorch.optim.LNSSGD(model_sgd.lns_parameters(), lr=0.1)
opt_madam = xltorch.optim.LNSMadam(model_mul.lns_parameters(), lr=0.05, use_pow=False)
opt_madam_pow = xltorch.optim.LNSMadam(model_mul_pow.lns_parameters(), lr=0.05, use_pow=True)
loss_fn = torch.nn.MSELoss(reduction='mean')

n_steps = 20
loss_hist_sgd, loss_hist_mul, loss_hist_mul_pow = [], [], []

for step in range(1, n_steps + 1):

    opt_sgd.zero_grad()
    loss_sgd = loss_fn(model_sgd(inp), target)
    loss_sgd.backward()
    opt_sgd.step()

    opt_madam.zero_grad()
    loss_mul = loss_fn(model_mul(inp), target)
    loss_mul.backward()
    opt_madam.step()

    opt_madam_pow.zero_grad()
    loss_mul_pow = loss_fn(model_mul_pow(inp), target)
    loss_mul_pow.backward()
    opt_madam_pow.step()

    loss_hist_sgd.append(loss_sgd.item())
    loss_hist_mul.append(loss_mul.item())
    loss_hist_mul_pow.append(loss_mul_pow.item())

    print(f"Iter {step:02d} | loss_SGD={loss_sgd.item():.2e} | "
          f"loss_Madam={loss_mul.item():.2e} | "
          f"loss_Madam_pow={loss_mul_pow.item():.2e}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, n_steps + 1), loss_hist_sgd, label='LNSSGD (lr=0.1)',  marker='o')
plt.plot(range(1, n_steps + 1), loss_hist_mul, label='LNSMadam (lr=0.05)', marker='s')
plt.plot(range(1, n_steps + 1), loss_hist_mul_pow, label='LNSMadam (lr=0.05, pow)', marker='^')
plt.title("MSE loss vs. optimisation step")
plt.xlabel("Step")
plt.ylabel("MSE loss")
plt.yscale("log")
plt.grid(True, ls='--', alpha=0.5)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()