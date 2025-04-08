import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from itertools import permutations
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

matplotlib.use('TkAgg')


# FUNCIONES
# def animar(solucion, nodos, conectividades, masas, orientacion_triangulos, intervalo=4):
#     t = solucion.t
#     Y = solucion.y
#     X_t = Y[:2 * len(nodos), :].T.reshape(len(t), len(nodos), 2)
#     escalado = [m * 4 for m in masas]
#
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_xlim(min(x for x, y in nodos) - 5, max(x for x, y in nodos) + 5)
#     ax.set_ylim(min(y for x, y in nodos) - 5, max(y for x, y in nodos) + 5)
#     ax.set_aspect('equal')
#     ax.set_title('Estructura de Barras')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.grid(True)
#
#     # Conectividades
#     lineas = [ax.plot([], [], 'k-', lw=1)[0] for _ in conectividades]
#
#     # Nodos
#     puntos = ax.scatter([n[0] for n in nodos], [n[1] for n in nodos], s=escalado, color='blue', label='Nodos')
#
#     # Etiquetas
#     etiquetas = [ax.text(nodos[i][0], nodos[i][1], str(i + 1), fontsize=12, color='red', ha='right')
#                  for i in range(len(nodos))]
#
#     # Texto del tiempo
#     tiempo_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
#                           fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
#
#     # Parches para triángulos
#     triangulos = []
#     for a, b, c in orientacion_triangulos:
#         coords_ini = [nodos[a], nodos[b], nodos[c]]
#         area = obtener_area_triangulo(a, b, c, nodos)
#         patch = Polygon(coords_ini, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5)
#         ax.add_patch(patch)
#         triangulos.append(patch)
#
#     def update(frame):
#         coordenadas = X_t[frame]
#         x, y = coordenadas[:, 0], coordenadas[:, 1]
#
#         # Nodos
#         puntos.set_offsets(coordenadas)
#
#         # Barras
#         for idx, (i, j, _) in enumerate(conectividades):
#             lineas[idx].set_data([x[i], x[j]], [y[i], y[j]])
#
#         # Etiquetas
#         for i, etiqueta in enumerate(etiquetas):
#             etiqueta.set_position((x[i], y[i]))
#
#         # Triángulos actualizados
#         for i, (a, b, c) in enumerate(orientacion_triangulos):
#             tri_coords = [coordenadas[a], coordenadas[b], coordenadas[c]]
#             area = obtener_area_triangulo(a, b, c, coordenadas)
#             triangulos[i].set_xy(tri_coords)
#             triangulos[i].set_facecolor('lightblue' if area > 0 else 'red')
#             triangulos[i].set_alpha(0.3 if area > 0 else 1)
#
#         # Actualizar texto de tiempo
#         tiempo_text.set_text(f'Tiempo: {t[frame]:.2f} s')
#
#         return lineas + etiquetas + triangulos + [puntos, tiempo_text]
#
#     ani = FuncAnimation(fig, update, frames=len(t), interval=intervalo, blit=True)
#     plt.show()


def animar(solucion, nodos, conectividades, masas, orientacion_triangulos, intervalo=4):
    t = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(t), len(nodos), 2)
    escalado = [m * 4 for m in masas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Estructura de barras
    ax1.set_xlim(min(x for x, y in nodos) - 5, max(x for x, y in nodos) + 5)
    ax1.set_ylim(min(y for x, y in nodos) - 5, max(y for x, y in nodos) + 5)
    ax1.set_aspect('equal')
    ax1.set_title('Estructura de Barras')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)

    lineas = [ax1.plot([], [], 'k-', lw=1)[0] for _ in conectividades]

    puntos = ax1.scatter([n[0] for n in nodos], [n[1] for n in nodos], s=escalado, color='blue', label='Nodos')

    etiquetas = [ax1.text(nodos[i][0], nodos[i][1], str(i + 1), fontsize=12, color='red', ha='right')
                 for i in range(len(nodos))]

    tiempo_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                           fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    triangulos = []
    for a, b, c in orientacion_triangulos:
        coords_ini = [nodos[a], nodos[b], nodos[c]]
        area = obtener_area_triangulo(a, b, c, nodos)
        patch = Polygon(coords_ini, closed=True, facecolor='lightblue', edgecolor='blue', alpha=0.5)
        ax1.add_patch(patch)
        triangulos.append(patch)

    # Funcion Peso
    ax2.set_aspect('auto')
    ax2.set_title('P(t)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('P')
    ax2.grid(True)
    ax2.plot(t, P(t), color='black')
    Pt, = ax2.plot([], [], 'mo', markersize=10)

    def update(frame):
        coordenadas = X_t[frame]
        x, y = coordenadas[:, 0], coordenadas[:, 1]

        # Nodos
        puntos.set_offsets(coordenadas)

        # Barras
        for idx, (i, j, _) in enumerate(conectividades):
            lineas[idx].set_data([x[i], x[j]], [y[i], y[j]])

        # Etiquetas
        for i, etiqueta in enumerate(etiquetas):
            etiqueta.set_position((x[i], y[i]))

        # Triángulos actualizados
        for i, (a, b, c) in enumerate(orientacion_triangulos):
            tri_coords = [coordenadas[a], coordenadas[b], coordenadas[c]]
            area = obtener_area_triangulo(a, b, c, coordenadas)
            triangulos[i].set_xy(tri_coords)
            triangulos[i].set_facecolor('lightblue' if area > 0 else 'red')
            triangulos[i].set_alpha(0.3 if area > 0 else 1)

        # Actualizar texto de tiempo
        tiempo_text.set_text(f'Tiempo: {t[frame]:.2f} s')

        # Particula
        Pt.set_data([t[frame]], [P(t[frame])])

        return lineas + etiquetas + triangulos + [puntos, tiempo_text, Pt]

    ani = FuncAnimation(fig, update, frames=len(t), interval=intervalo, blit=True)
    plt.tight_layout()
    plt.show()

def obtener_masas(nodos, conectividades, rho):
    masas = np.zeros(len(nodos))

    for i in range(len(nodos)):
        for ni, nj, r in conectividades:
            if (ni == i or nj == i):
                masas[i] += ((np.linalg.norm(np.array(nodos[ni]) - np.array(nodos[nj])) * rho ) / 2)

    return masas.tolist()


def obtener_area_triangulo(a, b, c, nodos):
    A = np.array([*nodos[a], 0])
    B = np.array([*nodos[b], 0])
    C = np.array([*nodos[c], 0])

    AB = B - A
    AC = C - A

    z = np.cross(AB, AC)[2]
    return z/2


def obtener_orientacion_triangulos(nodos, conectividades):
    aristas = [(i, j) for i, j, _ in conectividades]

    triangulos = set()
    for a, b, c in combinations(range(len(nodos)), 3):
        if ((a, b) in aristas or (b, a) in aristas) and ((b, c) in aristas or (c, b) in aristas) and ((a, c) in aristas or (c, a) in aristas):
            triangulos.add(tuple(sorted((a, b, c))))

    triangulos_orientados = set()
    for t in triangulos:
        # combinaciones
        for a, b, c in permutations(t):
            if (obtener_area_triangulo(a, b, c, nodos) > 0):
                triangulos_orientados.add((a, b, c))
                break;

    return triangulos_orientados


# DATOS
rho = 0.5
E = 50
Area = 0.05
# x, y
nodos = [(10, 0),
         (15, 0),
         (30, 0),
         (10, 10),
         (15, 10),
         (0, 20),
         (10, 20),
         (30, 20)]
# nodo i, nodo j, rigidez
conectividades = [(0, 4, 3),
                  (4, 3, 9),
                  (1, 0, 3),
                  (4, 1, 3),
                  (0, 3, 3),
                  (3, 6, 9),
                  (6, 4, 9),
                  (6, 5, 9),
                  (3, 5, 9),
                  (6, 7, 9),
                  (4, 7, 9),
                  (7, 2, 0.3)]
masas = obtener_masas(nodos, conectividades, rho)
orientacion_triangulos = obtener_orientacion_triangulos(nodos, conectividades)
# posiciones iniciales + velocidades iniciales
Y0 = [coordenadas for nodo in nodos for coordenadas in nodo] + [0] * (2 * len(nodos))
intervalo = (0, 50)
t = np.linspace(*intervalo, 1000)
def P(t):
    return 0.1 + 0*t
# def P(t):
#     A = 1
#     f = 0.04
#     phi = 0
#
#     return A * np.sin(2 * np.pi * f * t + phi)


# PROCESO
def fuerza(x_i, x_j, x0_i, x0_j, k_ij):
    norm_x0 = np.linalg.norm(x0_j - x0_i)
    norm_x = np.linalg.norm(x_j - x_i)
    factor = 1 - (norm_x0 / norm_x) if norm_x != 0 else 0

    return k_ij * factor * (x_j - x_i)

def sistema_ecuaciones(t, Y):
    Y = np.array(Y)
    X = Y[:len(Y) // 2].reshape(-1, 2)
    V = Y[len(Y) // 2:].reshape(-1, 2)
    A = np.zeros_like(X)
    F = []

    # Calcula la fuerza neta en cada nodo
    for i, j, _ in conectividades:
        k_ij = E*Area/(np.linalg.norm(np.array(nodos[i]) - np.array(nodos[j])))
        F_ij = fuerza(X[i], X[j], np.array(nodos[i]), np.array(nodos[j]), k_ij)
        F.append(F_ij)

        A[i] += F_ij / masas[i]
        A[j] -= F_ij / masas[j]

    # Aplica Peso
    A[5][1] -= P(t) / masas[5]

    # Condiciones de borde
    V[0] = [0, 0]
    V[1][1] = 0
    A[0] = [0, 0]
    A[1][1] = 0

    return np.concatenate((V, A)).flatten()

Y = solve_ivp(sistema_ecuaciones, intervalo, Y0, t_eval=t)
animar(Y, nodos, conectividades, masas, orientacion_triangulos)
