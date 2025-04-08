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
def animar(solucion, nodos, conectividades, masas, orientacion_triangulos, intervalo=0):
    t = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(t), len(nodos), 2)
    escalado = [m * 8 for m in masas]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # Estructura de barras
    ax1.set_xlim(min(x for x, y in nodos) - 5, max(x for x, y in nodos) + 5)
    ax1.set_ylim(min(y for x, y in nodos) - 5, max(y for x, y in nodos) + 5)
    ax1.set_aspect('equal')
    ax1.set_title(f'Estructura de Barras {"Pequeñas" if pequeñas else "Grandes"} Deformaciones')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)

    colores_barra = ['black'] * len(conectividades)
    colores_barra[barra_a] = 'magenta'
    lineas = [ax1.plot([], [], '-k', lw=1.5, color=colores_barra[i])[0] for i in range(len(conectividades))]

    colores = ['blue'] * len(nodos)
    colores[nodo_b] = 'magenta'
    puntos = ax1.scatter([n[0] for n in nodos], [n[1] for n in nodos], s=escalado, color=colores, label='Nodos')

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

    # Evolucion nodo b
    xt = []
    yt = []
    ttt = []

    for ii, tt in enumerate(solucion.t):
        ttt.append(tt)
        xt.append(X_t[ii][nodo_b][0])

    ax3.set_aspect('auto')
    ax3.set_title(f'Nodo {nodo_b + 1} t vs x')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.grid(True)
    ax3.plot(ttt, xt, color='black')
    xt_b, = ax3.plot([], [], 'mo', markersize=10)

    for ii, tt in enumerate(solucion.t):
        yt.append(X_t[ii][nodo_b][1])

    ax4.set_aspect('auto')
    ax4.set_title(f'Nodo {nodo_b + 1} t vs y')
    ax4.set_xlabel('t')
    ax4.set_ylabel('y')
    ax4.grid(True)
    ax4.plot(ttt, yt, color='black')
    yt_b, = ax4.plot([], [], 'mo', markersize=10)

    ax5.set_aspect('auto')
    ax5.set_title(f'Nodo {nodo_b + 1} x(t) vs y(t)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.grid(True)
    ax5.plot(xt, yt, color='black')
    xyt_b, = ax5.plot([], [], 'mo', markersize=10)

    # Evolucion barra a
    ax6.set_aspect('auto')
    ax6.set_title(f'Tensión Barra {barra_a}')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.grid(True)

    # ax6.plot(xt, yt, color='black')

    def update(frame):
        coordenadas = X_t[frame]
        x, y = coordenadas[:, 0], coordenadas[:, 1]

        # Nodos
        puntos.set_offsets(coordenadas)

        # Barras
        for idx, (i, j) in enumerate(conectividades):
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

        # Nodo b
        xt_b.set_data([t[frame]], [X_t[frame][nodo_b][0]])
        yt_b.set_data([t[frame]], [X_t[frame][nodo_b][1]])
        xyt_b.set_data([X_t[frame][nodo_b][0]], [X_t[frame][nodo_b][1]])

        return lineas + etiquetas + triangulos + [puntos, tiempo_text, Pt, xt_b, yt_b, xyt_b]

    ani = FuncAnimation(fig, update, frames=len(t), interval=intervalo, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()


def obtener_masas(nodos, conectividades, rho):
    masas = np.zeros(len(nodos))

    for i in range(len(nodos)):
        for ni, nj in conectividades:
            if (ni == i or nj == i):
                masas[i] += ((np.linalg.norm(np.array(nodos[ni]) - np.array(nodos[nj])) * rho) / 2)

    return masas.tolist()


def obtener_area_triangulo(a, b, c, nodos):
    A = np.array([*nodos[a], 0])
    B = np.array([*nodos[b], 0])
    C = np.array([*nodos[c], 0])

    AB = B - A
    AC = C - A

    z = np.cross(AB, AC)[2]
    return z / 2


def obtener_orientacion_triangulos(nodos, conectividades):
    aristas = [(i, j) for i, j in conectividades]

    triangulos = set()
    for a, b, c in combinations(range(len(nodos)), 3):
        if ((a, b) in aristas or (b, a) in aristas) and ((b, c) in aristas or (c, b) in aristas) and (
                (a, c) in aristas or (c, a) in aristas):
            triangulos.add(tuple(sorted((a, b, c))))

    triangulos_orientados = set()
    for t in triangulos:
        # combinaciones
        for a, b, c in permutations(t):
            if (obtener_area_triangulo(a, b, c, nodos) > 0):
                triangulos_orientados.add((a, b, c))
                break;

    return triangulos_orientados


def obtener_rigidez(nodos, conectividades, E, A):
    k = []

    for i, j in conectividades:
        k_ij = E * A / (np.linalg.norm(np.array(nodos[i]) - np.array(nodos[j])))
        k.append(k_ij)

    return k


# DATOS
rho = 0.5
E = 50
A = 0.05
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
conectividades = [(0, 4),
                  (4, 3),
                  (1, 0),
                  (4, 1),
                  (0, 3),
                  (3, 6),
                  (6, 4),
                  (6, 5),
                  (3, 5),
                  (6, 7),
                  (4, 7),
                  (7, 2)]
masas = obtener_masas(nodos, conectividades, rho)
orientacion_triangulos = obtener_orientacion_triangulos(nodos, conectividades)
K = obtener_rigidez(nodos, conectividades, E, A)
# posiciones iniciales + velocidades iniciales
Y0 = [coordenadas for nodo in nodos for coordenadas in nodo] + [0] * (2 * len(nodos))
intervalo = (0, 50)
t = np.linspace(*intervalo, 1000)
nodo_b = 7  # nodo 8
barra_a = 9  # barra 10
pequeñas = True


def P(t):
    return 0.1 + 0 * t


# def P(t):
#     A = 0.5
#     f = 0.05
#     phi = 0
#
#     return A * np.sin(2 * np.pi * f * t + phi)

# PROCESO


def fuerza_pequeñas_deformaciones(x_i, x_j, x0_i, x0_j, k_ij):
    norm_0 = np.linalg.norm(x0_j - x0_i)
    norm = np.linalg.norm(x_j - x_i)

    return k_ij * (1 - (norm_0 / norm)) * (x_j - x_i)


def fuerza_grandes_deformaciones(x_i, x_j, x0_i, x0_j, k_ij):
    norm_0 = np.linalg.norm(x0_j - x0_i)
    norm = np.linalg.norm(x_j - x_i)

    return k_ij * ((norm / norm_0) - 1) * (x0_j - x0_i)


def sistema_ecuaciones(t, Y):
    Y = np.array(Y)
    X = Y[:len(Y) // 2].reshape(-1, 2)
    V = Y[len(Y) // 2:].reshape(-1, 2)
    A = np.zeros_like(X)
    F = []

    a = 0
    # Calcula la fuerza neta en cada nodo
    for i, j in conectividades:
        if (pequeñas):
            F_ij = fuerza_pequeñas_deformaciones(X[i], X[j], np.array(nodos[i]), np.array(nodos[j]), K[a])
            F.append(F_ij)

            A[i] += F_ij / masas[i]
            A[j] -= F_ij / masas[j]
        else:
            F_ij = fuerza_grandes_deformaciones(X[i], X[j], np.array(nodos[i]), np.array(nodos[j]), K[a])
            F.append(F_ij)

            A[i] += F_ij / masas[i]
            A[j] -= F_ij / masas[j]

        a += 1

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
