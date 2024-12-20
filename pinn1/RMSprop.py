import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Параметры
L = 1
T = 10
alpha = 0.01
learning_rate = 0.0001
epochs = 5000

N_collocation = 100
N_initial = 20
N_boundary = 20

# Коллокационные точки
x_col = torch.linspace(0, L, N_collocation)
t_col = torch.linspace(0, T, N_collocation)
x_col, t_col = torch.meshgrid(x_col, t_col, indexing='ij')
x_col = x_col.flatten().unsqueeze(1)
t_col = t_col.flatten().unsqueeze(1)
X_col = torch.cat((x_col, t_col), dim=1).to(device)

# Начальное условие
x_initial = torch.linspace(0, L, N_initial).unsqueeze(1)
t_initial = torch.zeros(N_initial, 1)
X_initial = torch.cat((x_initial, t_initial), dim=1).to(device)
u_initial = torch.sin(torch.pi * x_initial).to(device)

# Граничные условия
x_boundary_left = torch.zeros(N_boundary, 1)
t_boundary_left = torch.linspace(0, T, N_boundary).unsqueeze(1)
X_boundary_left = torch.cat((x_boundary_left, t_boundary_left), dim=1).to(device)
u_boundary_left = torch.zeros(N_boundary, 1).to(device)

x_boundary_right = torch.full((N_boundary, 1), L)
t_boundary_right = torch.linspace(0, T, N_boundary).unsqueeze(1)
X_boundary_right = torch.cat((x_boundary_right, t_boundary_right), dim=1).to(device)
u_boundary_right = torch.zeros(N_boundary, 1).to(device)

# Комбинируем
X_boundary = torch.cat((X_boundary_left, X_boundary_right), dim=0)
u_boundary = torch.cat((u_boundary_left, u_boundary_right), dim=0)


class ThermalCond(nn.Module):
    def __init__(self):
        super(ThermalCond, self).__init__()
        # Задаем архитектуру: 3 скрытых слоя по 20 нейронов с функцией активации Tanh
        self.layers = nn.Sequential(
            nn.Linear(2, 20), # На вход x и t
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)) # На выход U(x,t)

    def forward(self, x):
        return self.layers(x)


model = ThermalCond().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

loss_history = []

# Обучаем с прогресс баром
pbar = tqdm(range(epochs), desc='Training', ncols=80)
for epoch in pbar:
    optimizer.zero_grad()

    # Loss для начальных условий
    u_pred_initial = model(X_initial)
    loss_initial = mse_loss(u_pred_initial, u_initial)

    # Loss для граничных условий
    u_pred_boundary = model(X_boundary)
    loss_boundary = mse_loss(u_pred_boundary, u_boundary)

    # Loss для точек уравнения
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)
    X_col = torch.cat((x_col, t_col), dim=1).to(device)
    u_pred_col = model(X_col)
    grads = torch.autograd.grad(u_pred_col, [x_col, t_col],
        grad_outputs=torch.ones_like(u_pred_col),
        create_graph=True, retain_graph=True, allow_unused=True)
    du_dx = grads[0]
    du_dt = grads[1]
    d2u_dx2 = torch.autograd.grad(du_dx, x_col,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True, retain_graph=True)[0]
    residual = du_dt - alpha * d2u_dx2
    loss_pde = mse_loss(residual, torch.zeros_like(residual))

    # Итоговая Loss
    loss = loss_initial + loss_boundary + 100*loss_pde
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()

    # Обновление отображения tqdm
    pbar.set_postfix({'Loss': f'{loss.item():.2e}'})

    # if epoch % 1000 == 0:
    #     print(f'Epoch {epoch}, Loss: {loss.item()}')



# Чертежи
x_test = torch.linspace(0, L, 100).unsqueeze(1)
t_test = torch.linspace(0, T, 100).unsqueeze(1)
X_test = torch.cat((x_test, t_test), dim=1).to(device)
u_pred = model(X_test).detach().to(device)

# Аналитическое решение
u_exact = torch.sin(torch.pi * x_test) * torch.exp(- alpha * torch.pi**2 * t_test)

# График аналитического и PINN решений
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_test, u_pred, label='PINN Solution')
plt.plot(x_test, u_exact, '--', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title(f't = {T}')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Зависимость Loss от Epochs
plt.figure(figsize=(8, 4), dpi=250)
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel('Эпохи')
plt.ylabel('Loss (MSE)')
plt.title('Функция потерь во время обучения (MSE)')
plt.grid(True)
plt.show()

# График в 3д: U(x,t)
x_plot = torch.linspace(0, L, 100)
t_plot = torch.linspace(0, T, 100)
X_plot, T_plot = torch.meshgrid(x_plot, t_plot, indexing='xy')
inputs_plot = torch.cat((X_plot.flatten().unsqueeze(1), T_plot.flatten().unsqueeze(1)), dim=1).to(device)

with torch.no_grad():
    u_pred_plot = model(inputs_plot).to(device)

U_plot = u_pred_plot.numpy().reshape(X_plot.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_plot.numpy(), T_plot.numpy(), U_plot, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
ax.set_title('Зависимость U(x,t)')

plt.show()
