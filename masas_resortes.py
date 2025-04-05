import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
# Definimos las masas de las partículas
masas = [0.5, 0.5, 2, 0.5, 2]

# Posiciones iniciales
X_0 = np.array([
    [1, 1],       # Partícula 1
    [10, 0.5],    # Partícula 2
    [6, 4],       # Partícula 3
    [9, 6],       # Partícula 4
    [2, 5]        # Partícula 5
])

# Velocidades iniciales
vel_iniciales = np.zeros((5, 2))  # Todas las velocidades iniciales en 0

# Condiciones iniciales aplanadas
Y0 = np.hstack((X_0.flatten(), vel_iniciales.flatten()))

# Otros parámetros
k_ij = 20  # Constante de rigidez del resorte ()
W = 5      # Fuerza externa
interv = (0, 50)  # Intervalo de tiempo
t = np.linspace(*interv, 1000)  # Tiempo para evaluar la solución

# Función para calcular el factor del resorte
def factor(x_i, x_j, x0_i, x0_j):
    norm_x0 = np.linalg.norm(x0_j - x0_i)
    norm_x = np.linalg.norm(x_j - x_i)
    return 1 - (norm_x0 / norm_x) if norm_x != 0 else 0  # Evitar división por cero

# Fuerzas en x e y
def Fx(x_i, x_j, x0_i, x0_j):
    return k_ij * factor(x_i, x_j, x0_i, x0_j) * (x_j[0] - x_i[0])

def Fy(x_i, x_j, x0_i, x0_j):
    return k_ij * factor(x_i, x_j, x0_i, x0_j) * (x_j[1] - x_i[1])

# Sistema de ecuaciones diferenciales
def f(t, Yv):
    # Separar posiciones y velocidades
    X = Yv[:10].reshape((5, 2))  # Primeras 10 entradas son las posiciones
    velocidades = Yv[10:].reshape((5, 2))  # Últimas 10 entradas son las velocidades

    # Inicializar aceleraciones
    Y = np.zeros_like(X)

    # Posiciones iniciales de referencia (conexión inicial)
    x1_0 = X_0[0]
    Y[0] = 0
    Y[0, 1] = 0
    Y[4] = 0
    # Cálculo de aceleraciones específicas
    Y[1] += (1 / masas[1]) * (
        -Fx(x1_0, X[1], X_0[0], X_0[1])
        + Fx(X[1], X[2], X_0[1], X_0[2])
        + Fx(X[1], X[3], X_0[1], X_0[3])
    )
    Y[1, 1] += (1 / masas[1]) * (
        -Fy(x1_0, X[1], X_0[0], X_0[1])
        + Fy(X[1], X[2], X_0[1], X_0[2])
        + Fy(X[1], X[3], X_0[1], X_0[3])
    )

    Y[2] += (1 / masas[2]) * (
        -Fx(x1_0, X[2], X_0[0], X_0[2])
        - Fx(X[1], X[2], X_0[1], X_0[2])
        + Fx(X[2], X[3], X_0[2], X_0[3])
        + Fx(X[2], X[4], X_0[2], X_0[4])
    )
    Y[2, 1] += (1 / masas[2]) * (
        -Fy(x1_0, X[2], X_0[0], X_0[2])
        - Fy(X[1], X[2], X_0[1], X_0[2])
        + Fy(X[2], X[3], X_0[2], X_0[3])
        + Fy(X[2], X[4], X_0[2], X_0[4])
    )

    Y[3] += (1 / masas[3]) * (
        -Fx(X[1], X[3], X_0[1], X_0[3])
        - Fx(X[2], X[3], X_0[2], X_0[3])
        + Fx(X[3], X[4], X_0[3], X_0[4])
    )
    Y[3, 1] += (1 / masas[3]) * (
        -Fy(X[1], X[3], X_0[1], X_0[3])
        - Fy(X[2], X[3], X_0[2], X_0[3])
        + Fy(X[3], X[4], X_0[3], X_0[4])
        - W
    )

    Y[4, 1] += (1 / masas[4]) * (
        -Fy(x1_0, X[4], X_0[0], X_0[4])
        - Fy(X[2], X[4], X_0[2], X_0[4])
        - Fy(X[3], X[4], X_0[3], X_0[4])
    )

    # Aplanar velocidades y aceleraciones para devolver a solve_ivp
    return np.hstack((velocidades.flatten(), Y.flatten()))

# Resolver el sistema
sol = solve_ivp(f, interv, Y0, t_eval=t)
# Definir conexiones entre masas
conexiones = [(0,1), (0,2), (0,4), (1,2), (1,3), (2,3), (2,4), (3,4)]

fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
points, = ax.plot([], [], 'bo')
line_segments = [ax.plot([], [], 'k-')[0] for _ in conexiones]

def init():
    points.set_data([], [])
    for line in line_segments:
        line.set_data([], [])
    return [points] + line_segments

def update(frame):
    X = sol.y[:10, frame].reshape((5, 2))
    points.set_data(X[:, 0], X[:, 1])
    for line, (i, j) in zip(line_segments, conexiones):
        line.set_data([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]])
    return [points] + line_segments

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
ani.save("animacion.gif", writer=PillowWriter(fps=30))
plt.show()

# Graficar las posiciones de cada partícula
for i in range(5):
    plt.figure(figsize=(10, 4))
    # Posiciones x
    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[i * 2, :], label=f'Partícula {i+1} - x')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición x')
    plt.legend()
    plt.grid(True)

    # Posiciones y
    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[i * 2 + 1, :], label=f'Partícula {i+1} - y')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Posición y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
