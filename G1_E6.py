import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import matplotlib.widgets as widgets

# Parámetros del sistema
m = 0.5  # Masa
k_ij = 20  # Constante de rigidez del resorte
A = 5  # Amplitud de la excitación sinusoidal
f = 1  # Frecuencia en Hz (se puede modificar para experimentar)
omega = 2 * np.pi * f  # Frecuencia angular
interv = (0, 10)  # Intervalo de tiempo
num_steps = 500

t = np.linspace(*interv, num_steps)  # Tiempo para evaluar la solución

# Posiciones iniciales
X_0 = np.array([
    [0, 0], [0, 5], [5, 0], [5, 5], [10, 0], [10, 5], [15, 0]
])
connections = [(0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 6), (1, 2), (3, 2), (3, 4), (5, 4)]

# Velocidades iniciales en 0
vel_iniciales = np.zeros((7, 2))
Y0 = np.hstack((X_0.flatten(), vel_iniciales.flatten()))

# Función para calcular la fuerza de resorte
def force(x_i, x_j, x0_i, x0_j):
    norm_x0 = np.linalg.norm(x0_j - x0_i)
    norm_x = np.linalg.norm(x_j - x_i)
    factor = 1 - (norm_x0 / norm_x) if norm_x != 0 else 0
    return k_ij * factor * (x_j - x_i)

# Sistema de ecuaciones diferenciales con fuerza sinusoidal
def f(t, Yv):
    X = Yv[:14].reshape((7, 2))
    V = Yv[14:].reshape((7, 2))
    acel  = np.zeros_like(X)
    
    for i, j in connections:
        F = force(X[i], X[j], X_0[i], X_0[j])
        acel [i] += F / m
        acel [j] -= F / m

    # **Aplicar fuerza sinusoidal en la masa 7**
    acel [6, 1] -= (A * np.sin(omega * t)) / m  

    # **Mantener fijas las masas empotradas (1 y 2)**
    acel [0] = np.array([0, 0])
    acel [1] = np.array([0, 0])
    V[0] = np.array([0, 0])
    V[1] = np.array([0, 0])

    return np.hstack((V.flatten(), acel .flatten()))

# Resolver el sistema
t_eval = np.linspace(*interv, num_steps)
sol = solve_ivp(f, interv, Y0, t_eval=t_eval)
trajectories = sol.y[:14, :].reshape(7, 2, -1).transpose(2, 0, 1)

# Animación
fig, ax = plt.subplots()
ax.set_xlim(-2, 17)
ax.set_ylim(-10, 7)
points, = ax.plot([], [], 'bo', markersize=8)
lines, = ax.plot([], [], 'k-', lw=2)

def init():
    points.set_data([], [])
    lines.set_data([], [])
    return points, lines

def update(frame):
    X_frame = trajectories[frame]
    points.set_data(X_frame[:, 0], X_frame[:, 1])
    line_x, line_y = [], []
    for i, j in connections:
        line_x.extend([X_frame[i, 0], X_frame[j, 0], None])
        line_y.extend([X_frame[i, 1], X_frame[j, 1], None])
    lines.set_data(line_x, line_y)
    return points, lines

# Botón de inicio
def start_animation(event):
    if not ani.event_source.is_running():  # Solo inicia si está detenida
        ani.event_source.start()

ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = widgets.Button(ax_button, 'Iniciar')
button.on_clicked(start_animation)

# Configurar la animación
dt = 0.01  # Paso de tiempo
ani = animation.FuncAnimation(fig, update, frames=500, interval=dt*1000, blit=True)
ani.event_source.stop()  # Asegurar que inicie detenida

plt.show()
