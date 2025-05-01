import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from itertools import permutations
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from sympy import primenu

matplotlib.use('TkAgg')


# FUNCIONES
def animar(solucion, nodos, conectividades, masas, orientacion_triangulos, F_b, intervalo=0):
    t = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(t), len(nodos), 2)

    escalado = [m * 80 for m in masas]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # Poner en True para pantalla completa
    if False:
        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()
        except AttributeError:
            try:
                manager.window.state('zoomed')  # Para TkAgg
            except AttributeError:
                pass  # Si no se puede, no pasa nada
    
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # Estructura de barras
    ax1.set_xlim(min(x for x, y in nodos) - 7, max(x for x, y in nodos) + 7)
    ax1.set_ylim(min(y for x, y in nodos) - 15, max(y for x, y in nodos) + 7)
    ax1.set_aspect('equal')
    ax1.set_title(f'Estructura de Barras {"Pequeñas" if pequeñas_deformaciones else "Grandes"} Deformaciones - {"Carga constante" if not carga_sinusoidal else "Carga sinusoidal"} ')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)

    colores_barra = ['black'] * len(conectividades)
    # Calculos para barra especial a
    colores_barra[barra_a] = 'magenta'
    i, j = conectividades[barra_a]
    x_barra_a_inicial_i, y_barra_a_inicial_i = nodos[i]
    x_barra_a_inicial_j, y_barra_a_inicial_j = nodos[j]
    distancia_inicial_barra_a = np.hypot(x_barra_a_inicial_j - x_barra_a_inicial_i, y_barra_a_inicial_j - y_barra_a_inicial_i)
    
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
    ax2.plot(t, P(t, carga_sinusoidal), color='black')
    Pt, = ax2.plot([], [], 'mo', markersize=10)

    # Evolucion nodo b
    xt = []
    yt = []
    ttt = []

    for ii, tt in enumerate(solucion.t):
        ttt.append(tt)
        xt.append(X_t[ii][nodo_b][0])

    ax3.set_aspect('auto')
    ax3.set_title(f'Nodo {nodo_b + 1} posición en x')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.grid(True)
    ax3.plot(ttt, xt, color='black')
    xt_b, = ax3.plot([], [], 'mo', markersize=10)

    for ii, tt in enumerate(solucion.t):
        yt.append(X_t[ii][nodo_b][1])

    ax4.set_aspect('auto')
    ax4.set_title(f'Nodo {nodo_b + 1} posición en y')
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
    ax6.set_title(f'Tensión Barra {barra_a + 1}')
    ax6.set_xlabel('t')
    ax6.set_ylabel('T')
    ax6.grid(True)
    ax6.plot(t, F_b, color='black')
    f_b, = ax6.plot([], [], 'mo', markersize=10)

    def update(frame):
        coordenadas = X_t[frame]
        x, y = coordenadas[:, 0], coordenadas[:, 1]

        # Nodos
        puntos.set_offsets(coordenadas)

        # Barras
        for idx, (i, j) in enumerate(conectividades):
            lineas[idx].set_data([x[i], x[j]], [y[i], y[j]])
        
        # Cambiar color de la barra especial (barra_a) según distancia
        i_barra_a, j_barra_a = conectividades[barra_a]
        dist_actual = np.hypot(x[j_barra_a] - x[i_barra_a], y[j_barra_a] - y[i_barra_a])
        delta = dist_actual - distancia_inicial_barra_a  
        delta_normalizado = np.clip(delta, -1, 1)
        if delta_normalizado >= 0:
            r = 0
            g = int(255 * delta_normalizado) 
            b = 0
        else:
            r = int(255 * (-delta_normalizado)) 
            g = 0
            b = 0
        color_hex = f'#{r:02x}{g:02x}{b:02x}'
        lineas[barra_a].set_color(color_hex)

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
        Pt.set_data([t[frame]], [P(t[frame], carga_sinusoidal)])

        # Nodo b
        xt_b.set_data([t[frame]], [X_t[frame][nodo_b][0]])
        yt_b.set_data([t[frame]], [X_t[frame][nodo_b][1]])
        xyt_b.set_data([X_t[frame][nodo_b][0]], [X_t[frame][nodo_b][1]])

        # Barra a
        f_b.set_data([t[frame]], [F_b[frame]])

        return lineas + etiquetas + triangulos + [puntos, tiempo_text, Pt, xt_b, yt_b, xyt_b, f_b]

    ani = FuncAnimation(fig, update, frames=len(t), interval=intervalo, blit=True, repeat=False)
    if generar_gif:
        tipo_deformacion = "Pequeñas deformaciones" if pequeñas_deformaciones else "Grandes deformaciones"
        tipo_carga = "Carga sinusoidal" if carga_sinusoidal else "Carga constante"
        nombre_archivo = f'{tipo_deformacion}-{tipo_carga}.gif'
        ani.save(nombre_archivo, writer='pillow', fps=20)
    plt.tight_layout()
    plt.show()


def obtener_masas(nodos, conectividades, rho):
    masas = np.zeros(len(nodos))

    for i in range(len(nodos)):
        for ni, nj in conectividades:
            if (ni == i or nj == i):
                masas[i] += ((np.linalg.norm(np.array(nodos[ni]) - np.array(nodos[nj])) * rho) / 2) / 20

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
        L = (np.linalg.norm(np.array(nodos[i]) - np.array(nodos[j])))
        k_ij = E * A / L
        k.append(k_ij)

    return k


def obtener_tiempo_desestabilizacion(solucion, orientacion_triangulos, nodos):
    T = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(T), len(nodos), 2)

    for i in range(len(T)):
        coordenadas = X_t[i]

        for a, b, c in orientacion_triangulos:
            area = obtener_area_triangulo(a, b, c, coordenadas)

            if area < 0:
                return T[i]

    return -1

def P(t, sinusoidal = False):
    
    if sinusoidal:
        A = 0.1
        f = 0.05
        phi = 0
        return A * np.sin(-2 * np.pi * f * t + phi)
    else:
        return 0.1 + t * 0


def fuerza_grandes_deformaciones(x_i, x_j, x0_i, x0_j, k_ij):
    norm_0 = np.linalg.norm(x0_j - x0_i)
    norm = np.linalg.norm(x_j - x_i)

    return k_ij * (1 - (norm_0 / norm)) * (x_j - x_i)


def fuerza_pequeñas_deformaciones(x_i, x_j, x0_i, x0_j, k_ij):
    norm_0 = np.linalg.norm(x0_j - x0_i)
    norm = np.linalg.norm(x_j - x_i)

    return k_ij * ((norm / norm_0) - 1) * (x0_j - x0_i)


def obtener_fuerzas_barra(solucion, nodos, barra, conectividades, K, A):
    T = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(T), len(nodos), 2)
    nodo_a = conectividades[barra][0]
    nodo_b = conectividades[barra][1]

    x0_i = nodos[nodo_a]
    x0_j = nodos[nodo_b]

    F = []

    for i in range(len(T)):
        coordenadas = X_t[i]

        x_i = coordenadas[nodo_a]
        x_j = coordenadas[nodo_b]

        F_ij = [0, 0]

        if (pequeñas_deformaciones):
            F_ij = fuerza_pequeñas_deformaciones(x_i, x_j, np.array(x0_i), np.array(x0_j), K[barra])
        else:
            F_ij = fuerza_grandes_deformaciones(x_i, x_j, np.array(x0_i), np.array(x0_j), K[barra])
            
        # print(f'Barra con coordenadas {x_i} y {x_j}. Fuerza: {F_ij}. Producto: {np.dot((x_j - x_i), F_ij)}')
        F_ij_signed = np.linalg.norm(F_ij) * np.sign(np.dot((x_i - x_j), F_ij))
        F.append(F_ij_signed/A)

    return F


def obterner_max_desplazamiento(solucion, nodos, t_final):
    T = solucion.t
    Y = solucion.y
    X_t = Y[:2 * len(nodos), :].T.reshape(len(T), len(nodos), 2)

    t_max = 0
    d_max = 0
    node = 0

    for i in range(len(T)):
        coordenadas = X_t[i]

        for j, nodo in enumerate(nodos):

            norm = np.linalg.norm(np.array(nodo) - coordenadas[j])

            if norm > d_max:
                d_max = norm
                t_max = T[i]
                node = j

        if T[i] == t_final:
            break

    print(f'Desplazamiento maximo: {d_max}')
    print(f'Tiempo de ocurrencia: {t_max}')
    print(f'Nodo: {node + 1}')


def sistema_ecuaciones(t, Y):
    Y = np.array(Y)
    X = Y[:len(Y) // 2].reshape(-1, 2)
    V = Y[len(Y) // 2:].reshape(-1, 2)
    A = np.zeros_like(X)

    # Calcula la fuerza neta en cada nodo
    for index, (i, j) in enumerate(conectividades):
        if (pequeñas_deformaciones):
            F_ij = fuerza_pequeñas_deformaciones(X[i], X[j], np.array(nodos[i]), np.array(nodos[j]), K[index])

            # if (barra_a == index):
            #     F_b.append(F_ij)

            A[i] += F_ij / masas[i]
            A[j] -= F_ij / masas[j]
        else:
            F_ij = fuerza_grandes_deformaciones(X[i], X[j], np.array(nodos[i]), np.array(nodos[j]), K[index])

            # if (barra_a == index):
            #     F_b.append(F_ij)

            A[i] += F_ij / masas[i]
            A[j] -= F_ij / masas[j]

    # Aplica Peso
    A[5][1] -= P(t, carga_sinusoidal) / masas[5]

    # Condiciones de borde
    # Nodo 1 no se mueve en x ni en y por lo tanto velocidad 0 en x e y
    V[0] = [0, 0]
    # Nodo 2 si se mueve en x y no en y por lo tanto velocidad en y 0 y en x es la calculada
    V[1][1] = 0
    # Nodo 3 no se mueve en x ni en y por lo tanto velocidad 0 en x e y
    V[2] = [0, 0]
    # Nodo 1, como la velocidad es 0, la aceleracion es 0 en x y en y
    A[0] = [0, 0]
    # Nodo 2. como la velociad en y es o, la aceleracion en y es 0. En x mantiene la calculada
    A[1][1] = 0
    # Nodo 3 como la velocidad es 0, la aceleracion es 0 en x y en y
    A[2] = [0, 0]

    return np.concatenate((V, A)).flatten()

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
t = np.linspace(*intervalo, 87)
nodo_b = 7  # nodo 8
barra_a = 9  # barra 10
pequeñas_deformaciones = False
carga_sinusoidal = False
generar_gif = False


# SOLUCION
Y = solve_ivp(sistema_ecuaciones, intervalo, Y0, method='RK23')
t_desestabilizacion = obtener_tiempo_desestabilizacion(Y, orientacion_triangulos, nodos)
F_b = obtener_fuerzas_barra(Y, nodos, barra_a, conectividades, K, A)
obterner_max_desplazamiento(Y, nodos, (50 if t_desestabilizacion == -1 else t_desestabilizacion))

print(f'Tiempo de desestabilización: {t_desestabilizacion}')

animar(Y, nodos, conectividades, masas, orientacion_triangulos, F_b)




