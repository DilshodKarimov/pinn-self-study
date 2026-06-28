import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Имя файла формата .csv
CSV_FILE = "data.csv"

# Значения параметров при генерации, для проверки качества модели
THETA_TRUE, MU_TRUE, SIGMA_TRUE = 0.7, 1.5, 0.3

# Количество вех для оптимизатора
ADAM_EPOCH = 30000

# Модель 

class PINN(nn.Module):
    def __init__(model):
        super().__init__()
        model.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        model.theta_pred = nn.Parameter(torch.tensor([0.1], dtype=torch.float32, requires_grad=True))
        model.mu_pred = nn.Parameter(torch.tensor([0.1], dtype=torch.float32, requires_grad=True))
        model.sigma_pred = nn.Parameter(torch.tensor([0.1], dtype=torch.float32, requires_grad=True))
        
    def forward(model, t):
        return model.net(t)

# Загрузка и обработка данных

try:
    table = pd.read_csv(CSV_FILE)
    print(f"Данные успешно загружены из {CSV_FILE}. Найдено строк: {len(table)}")
except FileNotFoundError:
    print(f"Ошибка! Файл {CSV_FILE} не найден. Сначала запустите generate_data_csv.py")
    exit()

t_data = torch.tensor(table['t'].values, dtype=torch.float32).view(-1, 1)
X_data = torch.tensor(table['X'].values, dtype=torch.float32).view(-1, 1)

# вектора точек 

T_MAX = t_data.max().item()
t_physics = torch.linspace(0, T_MAX, 200, dtype=torch.float32).view(-1, 1)


# Оптимизатор и функция loss

pinn = PINN()

optimizer_adam = torch.optim.Adam(list(pinn.parameters()), lr=0.001)

history_loss = []
history_theta, history_mu, history_sigma = [], [], []

def total_loss():
    '''
    # dX/dt напрямую
    X_pred_data = pinn(t_data)
    loss_data = torch.mean((X_pred_data - X_data) ** 2)
    t_physclone = t_physics.clone().detach().requires_grad_(True)
    X_phys = pinn(t_physclone)
    dX_dt = torch.autograd.grad
    (
        X_pred_phys, 
        t_physclone, 
        torch.ones_like(X_pred_phys), 
        create_graph=True
    )[0]
    res_X = (dX_dt - (pinn.theta_pred * (pinn.mu_pred - X_pred_phys)))**2 - (pinn.sigma_pred**2)
    loss_physics = torch.mean(res_X**2)
    return loss_data + 0.7*loss_physics
    '''
    # конечные разности
    X_pred_data = pinn(t_data)
    loss_data = torch.mean((X_pred_data - X_data) ** 2)
    dt = (t_physics[-1] - t_physics[0]) / (len(t_physics) - 1)
    X_phys = pinn(t_physics)
    X_current = X_phys[:-1]
    X_next = X_phys[1:]
    x_theta_mu = pinn.theta_pred * (pinn.mu_pred - X_current) * dt
    dx = X_next - X_current
    loss_x_theta_mu = torch.mean((dx - x_theta_mu) ** 2)
    res = dx - loss_x_theta_mu
    loss_sigma = torch.mean((res ** 2) - (pinn.sigma_pred ** 2) * dt) ** 2
    return loss_data + 0.7 * loss_x_theta_mu + 0.3 * loss_sigma

# Обучение

print("\n--- Начало обучения ---")
for epoch in range(ADAM_EPOCH+1):
    optimizer_adam.zero_grad()
    loss = total_loss()
    loss.backward()
    optimizer_adam.step()
    
    history_loss.append(loss.item())
    history_theta.append(pinn.theta_pred.item())
    history_mu.append(pinn.mu_pred.item())
    history_sigma.append(pinn.sigma_pred.item())
    
    if epoch % 500 == 0:
        err_theta = abs(pinn.theta_pred.item() - THETA_TRUE) / THETA_TRUE * 100
        err_mu = abs(pinn.mu_pred.item() - MU_TRUE) / MU_TRUE * 100
        err_sigma = abs(pinn.sigma_pred.item() - SIGMA_TRUE) / SIGMA_TRUE * 100
        print(f"Adam эпоха : {epoch:4d} | Loss: {loss.item():.6f} | Ошибка Theta: {err_theta:.2f}%, Ошибка Mu: {err_mu:.4f}%, Ошибка Sigma: {err_sigma:.4f}%")

print("\n--- Конец обучения ---")
print("\nИтоговые значения:")
print(f"Эпоха : {epoch:4d} | Loss: {loss.item():.6f} |  Theta: {pinn.theta_pred.item():.4f},  Mu: {pinn.mu_pred.item():.4f}, Sigma: {pinn.sigma_pred.item():.4f}")

# картинки

pinn.eval()
with torch.no_grad():
    X_pred_pinn = pinn(t_data).numpy()

plt.figure(figsize=(12, 5))
plt.plot(t_data.numpy(), X_data.numpy(), color='gray', alpha=0.6, label='Реальный X')
plt.plot(t_data.numpy(), X_pred_pinn, color='blue', linewidth=2.5, label='Предсказанный X')
plt.title('Реальный график СДУ и предсказанный')
plt.xlabel('Время (t)')
plt.ylabel('Значение (X)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')
plt.savefig('pinn_x(t).png', dpi=300, bbox_inches='tight')
print("График сохранен под именем pinn_x(t).png")

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(history_loss, color='red', label='Loss')
plt.yscale('log')
plt.title('Loss')
plt.grid(True, which="both", ls="--")
plt.subplot(2, 2, 2)
plt.plot(history_theta, label='theta предсказанная')
plt.axhline(y=THETA_TRUE, color='r', linestyle='--')
plt.title('График Theta')
plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot(history_mu, label='mu предсказанная', color='green')
plt.axhline(y=MU_TRUE, color='r', linestyle='--')
plt.title('График Mu')
plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(history_sigma, label='sigma предсказанная', color='blue')
plt.axhline(y=SIGMA_TRUE, color='r', linestyle='--')
plt.title('График Sigma')
plt.grid(True)
plt.tight_layout()
plt.savefig('pinn.png', dpi=300, bbox_inches='tight')
print("Графики сохранены под именем pinn.png")
plt.show()

