#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:19:09 2024

@author: Estudiante
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ejercicio 1
tabla_nutricional = pd.read_csv('datos/tabla_nutricional.csv', delimiter=';')
tabla_nutricional = tabla_nutricional.fillna(0)

alimentos = tabla_nutricional['Alimento']

# Ejercicio 2
frutas = ['Banana', 'Manzana', 'Tomate', 'Naranja', 'Mandarina', 'pera', 'Tomate envasado']

frutas_tabla = tabla_nutricional[tabla_nutricional['Alimento'].isin(frutas)]


verduras = ['Acelga', 'Zanahoria', 'Lechuga', 'Cebolla', 'Zapallo']
verduras_tabla = tabla_nutricional[tabla_nutricional['Alimento'].isin(verduras)]


dieta = {'hc': tabla_nutricional['HC (gr)'].sum(),
         'prot' : tabla_nutricional['Proteinas (gr)'].sum(),
         'grasas': tabla_nutricional['Grasas (gr)'].sum(),
         'sodio': tabla_nutricional['Na (mg)'].sum(),
         'fibra': tabla_nutricional['Fibra (gr)'].sum(),
         'frutas': frutas_tabla['Cantidad (gr/ml)'].sum(),
         'verduras': verduras_tabla['Cantidad (gr/ml)'].sum()}

def esta_dieta_balanceada(dieta):
    porcentaje_hc = dieta['hc'] / (dieta['hc'] + dieta['grasas'] + dieta['prot'])
    porcentaje_grasas = dieta['grasas'] / (dieta['hc'] + dieta['grasas'] + dieta['prot'])
    porcentaje_prot = dieta['prot'] / (dieta['hc'] + dieta['grasas'] + dieta['prot'])
    
    if ((porcentaje_hc < 55 or porcentaje_hc > 75) or 
        (porcentaje_grasas < 15 or porcentaje_grasas > 30) or 
        (porcentaje_prot < 10 or porcentaje_prot > 15)):
        return False
    
    if (dieta['sodio'] > 2000 or 
        dieta['fibra'] <= 25 or 
        dieta['frutas'] + dieta['verduras'] < 400):
        return False

    return True

esta_dieta_balanceada(dieta)

# Ejercicio 3

# Armo la matriz que representa el valor nutricional correspondiente a un gramo de cada alimento de la tabla nutricional
matriz_nutricional = tabla_nutricional.copy()
matriz_nutricional[['Na (mg)', 'Ca (mg)', 'Fe (mg)']] /= 1000
matriz_nutricional = matriz_nutricional.rename(columns={'Na (mg)': 'Na (g)', 'Ca (mg)': 'Ca (g)', 'Fe (mg)': 'Fe (g)'})
matriz_nutricional = matriz_nutricional.drop(columns=['Alimento'])
matriz_nutricional['Total (gr)'] = matriz_nutricional.sum(axis=1)
matriz_nutricional = matriz_nutricional.div(matriz_nutricional['Total (gr)'], axis=0)
matriz_nutricional = matriz_nutricional.drop(columns=['Cantidad (gr/ml)', 'Total (gr)'])
matriz_nutricional = matriz_nutricional.T


def conseguir_mat_norm_y_cent(matriz):
  """
  Consigue la matriz normalizada y centrada correspondiente a los datos
  independientes de todas las muestras

  Parámetros:
      matriz: matriz

  Devuelve:
      matriz_norm_y_cent: matriz normalizada y centrada
  """
  media = matriz.mean(axis=1)
  desv_estandar = matriz.std(axis=1, ddof=0)
  
  matriz_centrada = matriz.sub(media, axis=0)
  matriz_norm_y_cent = matriz_centrada.div(desv_estandar, axis=0)
  
  return matriz_norm_y_cent

def conseguir_componentes_principales(matriz):
    matriz_norm_y_cent = conseguir_mat_norm_y_cent(matriz)
    matriz_cov = (matriz_norm_y_cent @ matriz_norm_y_cent.T)/np.shape(matriz_nutricional)[1]
    
    # Consigo los avals y avecs de la matriz de covarianza y los ordeno de mayor a menor
    avals, avecs = np.linalg.eig(matriz_cov)
    index_avals = np.argsort(-avals)
    avals = avals[index_avals]
    avecs = avecs[:, index_avals]
    
    return avals, avecs


def pca(matriz, n):
  """
  Obtiene un conjunto de datos que explica la variable dependiente a partir
  de una combinación lineal de los datos originales donde se consideran los
  primeros n términos para cada observación.

  Parámetros:
      matriz: matriz de datos
      n: términos considerados

  Devuelve:
      proyeccion: matriz cuyas n columnas corresponden a la proyección de los puntos
      sobre las componentes principales
  """
  avecs = conseguir_componentes_principales(matriz)[1]
  n_avecs_principales = avecs[:,:n]
  proyeccion = matriz.T @ n_avecs_principales
  return proyeccion


avals, avecs = conseguir_componentes_principales(matriz_nutricional)
datos_proyectados_3 = pca(matriz_nutricional, 3)
plt.rcParams['figure.figsize'] = [10, 10]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(datos_proyectados_3.iloc[:, 0], datos_proyectados_3.iloc[:, 1], datos_proyectados_3.iloc[:, 2], color='red', marker='o')

plt.title('Proyección de los datos en tres dimensiones')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.legend() 


ax.set_xlim(-.3, .3)
ax.set_ylim(-.3, .3)
ax.set_zlim(-.3, .3)

plt.show()



# Ejercicio 4
alimentos_cl = ['Aceite girasol', 'Arroz', 'Azucar', 'Fideos secos', 'Harina trigo', 'Huevo', 'Pan Frances', 'Leche fluida entera', 'Yerba', 'Zanahoria', 'Tomate', 'Cebolla', 'Papa', 'Acelga', 'Naranja', 'Manzana', 'Bola de Lomo', 'Asado', 'Paleta ', 'Carne picada']
indices_alimentos = tabla_nutricional.index[tabla_nutricional['Alimento'].isin(alimentos_cl)]

matriz_nutricional = matriz_nutricional[indices_alimentos]

datos_proyectados_3 = pca(matriz_nutricional, 3)
plt.rcParams['figure.figsize'] = [10, 10]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(datos_proyectados_3.iloc[:, 0], datos_proyectados_3.iloc[:, 1], datos_proyectados_3.iloc[:, 2], color='red', marker='o')

plt.title('Proyección de los datos en tres dimensiones')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.legend()    


ax.set_xlim(-.2, .2)
ax.set_ylim(-.2, .2)
ax.set_zlim(-.2, .2)

plt.show()


# Ejercicio 5
tabla_cl = pd.read_csv('datos/consumidores_libres.csv', delimiter=';')

fechas = list(tabla_cl.columns.values[2:])

# Saco las berenjenas, ya que no están en la tabla nutricional
tabla_cl = tabla_cl[tabla_cl['PRODUCTOS'] != 'BERENJENAS']

# Renombro los alimentos para que queden igual que en la tabla nutricional
tabla_cl['PRODUCTOS'] = alimentos_cl

tabla_cl[fechas + ['Cantidad']] = tabla_cl[fechas + ['Cantidad']].div(tabla_cl['Cantidad'], axis=0)

tabla_nutricional_cl = tabla_nutricional.copy()
tabla_nutricional_cl = tabla_nutricional_cl[tabla_nutricional_cl['Alimento'].isin(alimentos_cl)]

nutrientes = list(tabla_nutricional_cl.columns.values[1:])

tabla_nutricional_cl = tabla_nutricional_cl.drop(columns=nutrientes[5:])

tabla_nutricional_cl[nutrientes] = tabla_nutricional_cl[nutrientes].div(tabla_nutricional_cl['Cantidad (gr/ml)'], axis=0)

