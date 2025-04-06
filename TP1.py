import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from itertools import permutations

matplotlib.use('TkAgg')


# FUNCIONES
def graficar(nodos, conectividades, masas, orientacion_triangulos):
    x_vals = [nodo[0] for nodo in nodos]
    y_vals = [nodo[1] for nodo in nodos]
    escalado = [m * 4 for m in masas]

    # Crear el gráfico
    plt.figure(figsize=(6, 6))

    # Graficar conexiones entre nodos
    for ni, nj, r in conectividades:
        x_line = [nodos[ni][0], nodos[nj][0]]
        y_line = [nodos[ni][1], nodos[nj][1]]
        plt.plot(x_line, y_line, 'k-', linewidth=1)  # Líneas negras

    # Pintar las áreas de los triángulos
    for a, b, c in orientacion_triangulos:
        x_tri = [nodos[a][0], nodos[b][0], nodos[c][0]]
        y_tri = [nodos[a][1], nodos[b][1], nodos[c][1]]

        if (obtener_area_tringulo(a, b, c, nodos) > 0):
            plt.fill(x_tri, y_tri, color='lightblue', alpha=0.6, edgecolor='blue')
        else:
            plt.fill(x_tri, y_tri, color='red', alpha=0.3, edgecolor='blue')

    # Graficar nodos
    plt.scatter(x_vals, y_vals, s=escalado, color='blue', label='Nodos')

    # Etiquetas de nodos
    for i, (x, y) in enumerate(nodos):
        plt.text(x, y, str(i + 1), fontsize=15, ha='right', color='red')

    # Configuración del gráfico
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Estructura de Barras')
    plt.grid(True)
    plt.legend()

    # Mostrar el gráfico
    plt.show()


def obtener_masas(nodos, conectividades):
    masas = np.zeros(len(nodos))

    for i in range(len(nodos)):
        for ni, nj, r in conectividades:
            if (ni == i or nj == i):
                masas[i] += (np.linalg.norm(np.array(nodos[ni]) - np.array(nodos[nj])) / 2)

    return masas.tolist()


def obtener_area_tringulo(a, b, c, nodos):
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
            if (obtener_area_tringulo(a, b, c, nodos) > 0):
                triangulos_orientados.add((a, b, c))
                break;

    return triangulos_orientados


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
masas = obtener_masas(nodos, conectividades)
orietacion_triangulos = obtener_orientacion_triangulos(nodos, conectividades)
# posiciones iniciales + velocidades iniciales
Y0 = [coordenadas for nodo in nodos for coordenadas in nodo] + [0] * (2 * len(nodos))
intervalo = (0, 50)
t = np.linspace(*intervalo, 1000)
def P(t):
    return 0.1
# def P(t):
#     A = 1
#     f = 2
#     phi = 0
#
#     return A * np.sin(2 * np.pi * f * t + phi)


# PROCESO
def obtener_rigidez(nodos, i, j):
    L = np.linalg.norm(np.array(nodos[i]) - np.array(nodos[j]))

    return E*A/L


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

    # for i, j, _ in conectividades:
    #     k_ij = E*A/(np.linalg.norm(np.array(nodos[i]) - np.array(nodos[j])))
    #     F = fuerza(X[i], X[j], nodos[i], nodos[j], k_ij)
    #     acel [i] += F / m
    #     acel [j] -= F / m

    print(X)
    print(V)
    print(A)

sistema_ecuaciones(t, Y0)
graficar(nodos, conectividades, masas, orietacion_triangulos)
