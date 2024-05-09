import numpy as np
import networkx as nx
import scipy.linalg

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
        line = f.readline()
        i = int(line.split()[0]) - 1
        j = int(line.split()[1]) - 1
        W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)

def descompLU(A):
    """
    Realiza la descomposición LU de una matriz

    Parámetros:
        A: Matriz cuadrada de tamaño nxn.

    Devuelve:
        L: Matriz triangular inferior
        U: Matriz triangular superior
    """
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()

    if m!=n:
        print('Matriz no cuadrada')
        return

    for i in range(m-2):
        pivot = Ac[i,i]
        for j in range(i+1,m):
            coc = -Ac[j,i] / pivot
            Ac[j,i:] = np.sum([Ac[j,i:],coc * Ac[i,i:]],axis=0)
            Ac[j,i] = -coc
    L = np.tril(Ac,-1) + np.eye(A.shape[0])
    U = np.triu(Ac)

    return L, U

def resolverLU(A, b):
    """
    Resuelve un sistema Ax = b por descomposición LU.

    Parámetros:
        A: Matriz cuadrada de tamaño nxn.
        b: Vector de tamaño nx1.

    Devuelve:
        x: Solución del sistema.
    """
    P, L, U = scipy.linalg.lu(A)
    y = scipy.linalg.solve_triangular(L,b,lower = True)
    x = scipy.linalg.solve_triangular(U,y)
    return x

def armar_matriz_diagonal(matriz_conectividad):
    """
    Arma la matriz D diagonal con los grados de la matriz de conectividad

    Parámetros:
        matriz_conectividad: Matriz de unos y ceros con diagonal de ceros

    Devuelve:
        matriz_diagonal: matriz D
    """
    n = np.shape(matriz_conectividad)[0]
    matriz_diagonal = np.zeros((n,n))

    c = matriz_conectividad.sum(axis=0)

    for j in range(n):
        if c[j] != 0:
            matriz_diagonal[j, j] = 1/c[j]
        else:
            matriz_diagonal[j, j] = 0
    return matriz_diagonal

def normalizar(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) 
    scr = np.zeros(npages) 

    D = armar_matriz_diagonal(M)
    R = M @ D
    I = np.eye(npages)
    e = np.ones((npages,))
    
    scr = normalizar(resolverLU(I - p*R, e))
    scr = scr/scr.sum(keepdims=1)
    rnk = np.flip(np.argsort(scr)) + 1
    return rnk, scr

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    
    return output
