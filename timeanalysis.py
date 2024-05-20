import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read result data

resultados_kmeans = pd.read_csv('kmeans_results.csv')
resultados_knn = pd.read_csv('knn_results.csv')

n_kmeans = resultados_kmeans['sample_size']
n_knn = resultados_knn['sample_size_train']
k1 = 2
k2 = 3
p = 29
m1 = 1
m2 = 1

factor1 = 1/1e8
factor2 = 1/1e8+1/1e9
factor3 = 1/1e8
factor4 = 1/1e8

iter1 = resultados_kmeans['iteraciones_2_clusters']
iter2 = resultados_kmeans['iteraciones_3_clusters']

complejidad_calculada1 = [n_kmeans[i]*k1*p*iter1[i]*factor1+0.005 for i in range(len(n_kmeans))]
complejidad_calculada2 = [n_kmeans[i]*k2*p*iter2[i]*factor2+0.006 for i in range(len(n_kmeans))]

comlpejidad_calculada_knn_lineal = [n_knn[i]*p*factor3+0.0006 for i in range(len(n_knn))]


tiempos_kmeans1 = resultados_kmeans['time_2_clusters']
tiempos_kmeans2 = resultados_kmeans['time_3_clusters']
tiempos_knn = resultados_knn['time_create_model']

#plotting



plt.plot(n_kmeans,complejidad_calculada1, label='KMeans con 2 clusters - Teórico')
plt.plot(n_kmeans,tiempos_kmeans1, label='KMeans con 2 clusters - Experimental')
plt.legend()
plt.title('Gráfico usando 2 Clusters en KMeans')
plt.xlabel('Tamaño de la Muestra')
plt.ylabel('Tiempo (s)')
plt.show()

plt.plot( n_kmeans,complejidad_calculada2, label='KMeans con 3 clusters - Teórico')
plt.plot(n_kmeans,tiempos_kmeans2, label='KMeans con 3 clusters - Experimental')
plt.legend()
plt.title('Gráfico usando 3 Clusters en KMeans')
plt.xlabel('Tamaño de la Muestra')
plt.ylabel('Tiempo (s)')
plt.show()

plt.plot( n_knn,comlpejidad_calculada_knn_lineal,  label='KNN - Teórico - Lineal')
plt.plot( n_knn,tiempos_knn,  label='KNN - Experimental')
plt.legend()
plt.title('Gráfico Creación Modelo KNN con datos de Prueba')
plt.xlabel('Tamaño de la Muestra')
plt.ylabel('Tiempo (s)')
plt.show()