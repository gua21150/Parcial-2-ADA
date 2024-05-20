import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import timeit
import matplotlib
matplotlib.use('TkAgg')  # Configura el backend antes de importar pyplot
import matplotlib.pyplot as plt

# Función para realizar clustering y medir tiempo
def perform_kmeans(data, n_clusters):
    start_time = timeit.default_timer()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    total_time = timeit.default_timer() - start_time
    iteraciones = kmeans.n_iter_
    score = silhouette_score(data, kmeans.labels_)
    return total_time, score, iteraciones

# Función para realizar KNN y medir tiempo
def perform_knn(x_train, x_test, y_train, y_test, n_neighbors):
    start_time = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    tempo1 = timeit.default_timer() - start_time  # tiempo de creación del modelo
    start_time = timeit.default_timer()
    y_pred = knn.predict(x_test)
    tempo2 = timeit.default_timer() - start_time  # tiempo únicamente de predicción de datos
    accuracy = accuracy_score(y_test, y_pred)
    return tempo1, tempo2, accuracy


if __name__ == '__main__':

    # Leer el archivo CSV
    file_path = 'transformed_data.csv'
    df = pd.read_csv(file_path)

    # Inicializar variables
    sample_size = 74  # se ha colocado que tenga un tamaño de 74 cada muestra para obtener al menos 30 entradas
    increment = 74
    num_rows = df.shape[0]

    # Estandarizacion del data frame
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df))

    # Lista para almacenar resultados
    resultsmeans = []

    # Iterar sobre el DataFrame incrementando el tamaño de la muestra
    while sample_size <= num_rows:
        # sample_indices = np.random.choice(range(num_rows), sample_size, replace=False)
        # sample = scaled_df[sample_indices]
        # sample = scaled_df.sample(n=sample_size, random_state=0)
        sample = scaled_df.iloc[0:sample_size]
        # KMeans con 2 clusters
        time_2_clusters, score_2_clusters, iteraciones_2_clusters = perform_kmeans(sample, 2)

        # KMeans con 3 clusters
        time_3_clusters, score_3_clusters, iteraciones_2_clusters = perform_kmeans(sample, 3)

        # Almacenar resultados
        resultsmeans.append({
            'sample_size': sample_size,
            'time_2_clusters': time_2_clusters,
            'score_2_clusters': score_2_clusters,
            'iteraciones_2_clusters': iteraciones_2_clusters,
            'time_3_clusters': time_3_clusters,
            'score_3_clusters': score_3_clusters,
            'iteraciones_3_clusters': iteraciones_2_clusters
        })

        # Incrementar tamaño de la muestra
        sample_size += increment

    # Convertir resultados a DataFrame
    results_df_kmeans = pd.DataFrame(resultsmeans)

    # Guardar resultados en un archivo CSV
    results_df_kmeans.to_csv('kmeans_results.csv', index=False)

    print("Resultados guardados en 'kmeans_results.csv'")

    """ Procedimiento para knn """
    # Separando los datos
    # La variable X no incluye la columna Response
    columns_to_exclude = ['Response']
    columns_to_select = [col for col in df.columns if col not in columns_to_exclude]

    X = df[columns_to_select]
    y = df["Response"]

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X))

    # Inicializar variables
    sample_size = 74  # se tiene que volver a iniciar en 74
    results_knn = []

    # Iterar sobre el DataFrame incrementando el tamaño de la muestra
    while sample_size <= num_rows:
        # Seleccionar una muestra del DataFrame
        # sample_indices = np.random.choice(range(num_rows), sample_size, replace=False)
        # X_sample = X_scaled[sample_indices]
        # y_sample = y[sample_indices]

        X_sample = X_scaled.iloc[0:sample_size]
        y_sample = y[0:sample_size]
        # Dividir la muestra en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=0)

        # KNN con 5 puntos vecinos
        time_create_model, time_apply_model, accuracy_5_knn = perform_knn(X_train, X_test, y_train, y_test, 5)

        # Almacenar resultados
        results_knn.append({
            'sample_size_train': sample_size*0.80,  # corresponde al 80%
            'time_create_model': time_create_model,
            'sample_size_test': sample_size*0.20,  # corresponde al 20%
            'time_apply_model': time_apply_model,
            'score_5_knn': accuracy_5_knn
        })

        # Incrementar tamaño de la muestra
        sample_size += increment

    # Convertir resultados a DataFrame
    results_df_knn = pd.DataFrame(results_knn)

    # Guardar resultados en un archivo CSV
    results_df_knn.to_csv('knn_results.csv', index=False)

    print("Resultados guardados en 'knn_results.csv'")

    """Graficar los resultados"""
    # Graficar kmeans 2
    plt.plot(results_df_kmeans['sample_size'], results_df_kmeans['time_2_clusters'])
    # Etiquetas y título del gráfico
    plt.xlabel('Entradas')
    plt.ylabel('Tiempo de corrida')
    plt.title('Gráfico usando 2 Clusters en KMeans')
    # Mostrar el gráfico
    plt.show()

    # Graficar kmeans 3
    plt.plot(results_df_kmeans['sample_size'], results_df_kmeans['time_3_clusters'])
    # Etiquetas y título del gráfico
    plt.xlabel('Entradas')
    plt.ylabel('Tiempo de corrida')
    plt.title('Gráfico usando 3 Clusters en KMeans')
    # Mostrar el gráfico
    plt.show()

    # Graficar knn creación modelo
    plt.plot(results_df_knn['sample_size_train'], results_df_knn['time_create_model'])
    # Etiquetas y título del gráfico
    plt.xlabel('Entradas datos prueba')
    plt.ylabel('Tiempo de corrida')
    plt.title('Gráfico Creación Modelo KNN con datos de Prueba')
    # Mostrar el gráfico
    plt.show()

    # Graficar knn 3 neighbors
    plt.plot(results_df_knn['sample_size_test'], results_df_knn['time_apply_model'])
    # Etiquetas y título del gráfico
    plt.xlabel('Entradas datos de testeo')
    plt.ylabel('Tiempo de corrida')
    plt.title('Gráfico Aplicando KNN en datos de Testeo')
    # Mostrar el gráfico
    plt.show()
