import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    Генерирует данные для СДУ и сохраняет их в файл .csv
        theta (float): коэффициент скорости возврата к среднему
        mu (float): среднее значение, кк оторому стремится x
        sigma (float): волатильность (коэффициент шума)
        x0 (float): x0
        T (float): общее время
        dt (float): шаг по времени
        T / H = количество точек
"""

THETA_TRUE, MU_TRUE, SIGMA_TRUE = 0.7, 1.5, 0.3
X0, T, DT = 3.0, 10.0, 0.01

# график

def plot_process(t, x):
    plt.figure(figsize=(10, 5))
    plt.plot(t, x)
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.title("СДУ Орнштейна-Уленбека")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Генератор

def generator(theta, mu, sigma, x0, T, dt, filename="data.csv"):
    
    N = int(T/dt)
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = x0
    
    # генерация СДУ Орнштейна-Уленбека по методу Эйлера-Маруямы 
    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal(0, 1)
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
        
    # массивы идут в pandas
    table = pd.DataFrame({
        't': t,
        'X': X
    })
    
    table.to_csv(filename, index=False)
    print(f"Успешно сгенерировано {N} точек. Данные сохранены в '{filename}'")
    plot_process(t, X)

if __name__ == "__main__":
    generator(THETA_TRUE, MU_TRUE, SIGMA_TRUE, X0, T, DT)
