# ============================================
# Primero: pip install -r requirements.txt
# Segundo: python main.py
# ============================================
# main.py - Estudio comparativo de clustering
# Dataset: Mall Customers
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import os

# --------------------------------------------
# Configuración global
# --------------------------------------------
RANDOM_STATE = 42
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["axes.grid"] = True


# ============================================
# 1. CARGA Y PREPROCESAMIENTO
# ============================================
def load_and_preprocess(path_csv: str):
    """
    Carga el dataset Mall Customers y realiza:
      - Eliminación de CustomerID
      - One-hot encoding de Gender/Genre
      - Estandarización de variables numéricas

    Devuelve:
      - df: DataFrame preprocesado
      - X_scaled: np.array con features escaladas
      - feature_names: lista de nombres de columnas usadas
    """
    df_raw = pd.read_csv(path_csv)
    df = df_raw.copy()

    # Eliminar columnas de ID si existen
    for col_id in ["CustomerID", "Customer Id", "customer_id"]:
        if col_id in df.columns:
            df = df.drop(columns=[col_id])
            print(f"[INFO] Columna ID '{col_id}' eliminada.")

    # One-hot encoding para género
    if "Gender" in df.columns:
        df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
        print("[INFO] One-hot encoding aplicado a 'Gender'.")
    elif "Genre" in df.columns:
        df = pd.get_dummies(df, columns=["Genre"], drop_first=True)
        print("[INFO] One-hot encoding aplicado a 'Genre'.")

    print("[INFO] Columnas después del preprocesamiento:")
    print(df.columns)

    feature_names = df.columns.tolist()
    X = df[feature_names].values

    # Estandarización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Shape de X_scaled:", X_scaled.shape)
    return df, X_scaled, feature_names


# ============================================
# 2. K-MEANS: ELBOW Y SILHOUETTE
# ============================================
def kmeans_elbow_silhouette(X_scaled, k_min=2, k_max=10):
    """
    Ejecuta K-Means para k en [k_min, k_max] y calcula:
      - WCSS / inercia (método del codo)
      - Silhouette score

    Devuelve:
      - dict con k_values, inertias, silhouettes
    """
    inertias = []
    silhouettes = []
    k_values = range(k_min, k_max + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)

    # Gráfico Elbow
    plt.figure()
    plt.plot(list(k_values), inertias, marker='o')
    plt.title("K-Means - Método del codo")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia (WCSS)")
    plt.tight_layout()
    plt.show()

    # Gráfico Silhouette
    plt.figure()
    plt.plot(list(k_values), silhouettes, marker='o')
    plt.title("K-Means - Coeficiente de silueta medio")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.show()

    print("[INFO] Silhouette por k:")
    for k, s in zip(k_values, silhouettes):
        print(f"  k = {k}: silhouette = {s:.4f}")

    results = {
        "k_values": list(k_values),
        "inertias": inertias,
        "silhouettes": silhouettes
    }
    return results


def run_kmeans_best_k(X_scaled, best_k):
    """
    Ejecuta K-Means con el K elegido y calcula métricas.
    Devuelve:
      - labels
      - modelo KMeans
      - métricas (silhouette, davies-bouldin)
    """
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    metrics = {
        "silhouette": sil,
        "davies_bouldin": db
    }
    return labels, kmeans, metrics


# ============================================
# 3. CLUSTERING JERÁRQUICO
# ============================================
def run_hierarchical_clustering(X_scaled, method="ward", metric="euclidean", max_clusters=5):
    """
    Ejecuta clustering jerárquico aglomerativo:
      - Calcula la matriz de enlace (linkage)
      - Dibuja el dendrograma
      - Corta el dendrograma para obtener 'max_clusters' clusters

    Devuelve:
      - labels
      - Z (matriz de enlace)
      - métricas (silhouette, davies-bouldin)
    """
    Z = linkage(X_scaled, method=method, metric=metric)

    # Dendrograma
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode='level', p=5)
    plt.title(f"Dendrograma - método={method}, métrica={metric}")
    plt.xlabel("Observaciones")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.show()

    labels = fcluster(Z, t=max_clusters, criterion='maxclust')

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    metrics = {
        "silhouette": sil,
        "davies_bouldin": db
    }
    return labels, Z, metrics


# ============================================
# 4. DBSCAN: BÚSQUEDA DE eps Y min_samples
# ============================================
def run_dbscan_grid(X_scaled, eps_values, min_samples_values):
    """
    Explora combinaciones de eps y min_samples para DBSCAN.
    Devuelve un DataFrame con resultados.
    """
    results = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)

            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)

            if n_clusters <= 1:
                sil = np.nan
                dbs = np.nan
            else:
                sil = silhouette_score(X_scaled, labels)
                dbs = davies_bouldin_score(X_scaled, labels)

            results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": sil,
                "davies_bouldin": dbs
            })

    results_df = pd.DataFrame(results)
    return results_df


def run_dbscan_best_params(X_scaled, eps, min_samples):
    """
    Ejecuta DBSCAN con parámetros elegidos y calcula métricas.
    Devuelve:
      - labels
      - modelo DBSCAN
      - métricas (silhouette, davies-bouldin, n_clusters, n_noise)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    if n_clusters <= 1:
        sil = np.nan
        dbs = np.nan
    else:
        sil = silhouette_score(X_scaled, labels)
        dbs = davies_bouldin_score(X_scaled, labels)

    metrics = {
        "silhouette": sil,
        "davies_bouldin": dbs,
        "n_clusters": n_clusters,
        "n_noise": n_noise
    }
    return labels, db, metrics


# ============================================
# 5. VISUALIZACIÓN CON PCA Y t-SNE
# ============================================
def plot_clusters_pca_tsne(X_scaled, labels_dict, title_prefix="Mall Customers"):
    """
    labels_dict: diccionario {nombre_modelo: etiquetas}
    Aplica PCA y t-SNE a 2D y dibuja scatter plots.
    """
    # PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    for name, labels in labels_dict.items():
        plt.figure()
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=40)
        plt.title(f"{title_prefix} - {name} (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.show()

    # t-SNE 2D (puede tardar un poco)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    for name, labels in labels_dict.items():
        plt.figure()
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", s=40)
        plt.title(f"{title_prefix} - {name} (t-SNE 2D)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.show()


# ============================================
# 6. FUNCIÓN PRINCIPAL
# ============================================
def main():
    # Ruta al CSV (ajusta si lo pones en otro sitio)
    data_path = os.path.join("data", "Mall_Customers.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"No se encontró el archivo en: {data_path}\n"
            f"Asegúrate de que 'Mall_Customers.csv' esté en la carpeta 'data'."
        )

    print("[INFO] Cargando y preprocesando datos...")
    df, X_scaled, feature_names = load_and_preprocess(data_path)

    # 1) K-Means: selección de K
    print("\n[INFO] Ejecutando K-Means para selección de K...")
    km_results = kmeans_elbow_silhouette(X_scaled, k_min=2, k_max=10)

    # Elegimos el K con mayor silhouette
    best_k_index = int(np.nanargmax(km_results["silhouettes"]))
    best_k = km_results["k_values"][best_k_index]
    print(f"[INFO] Mejor K según silhouette: {best_k}")

    labels_km, km_model, km_metrics = run_kmeans_best_k(X_scaled, best_k)
    print("\n[RESULTADOS] K-Means:")
    print(km_metrics)

    # 2) Clustering jerárquico
    print("\n[INFO] Ejecutando clustering jerárquico...")
    labels_h, Z, h_metrics = run_hierarchical_clustering(
        X_scaled, method="ward", metric="euclidean", max_clusters=best_k
    )
    print("\n[RESULTADOS] Jerárquico:")
    print(h_metrics)

    # 3) DBSCAN: grid search
    print("\n[INFO] Ejecutando DBSCAN (búsqueda en rejilla)...")
    eps_values = [0.3, 0.5, 0.8, 1.0, 1.2]
    min_samples_values = [3, 5, 10]

    dbscan_grid = run_dbscan_grid(X_scaled, eps_values, min_samples_values)
    print("\n[RESULTADOS] Grid DBSCAN:")
    print(dbscan_grid)

    # Selección de mejores hiperparámetros (según silhouette)
    valid_db = dbscan_grid.dropna(subset=["silhouette"])
    if not valid_db.empty:
        best_idx = valid_db["silhouette"].idxmax()
        best_eps = valid_db.loc[best_idx, "eps"]
        best_ms = int(valid_db.loc[best_idx, "min_samples"])
        print("\n[INFO] Mejores parámetros DBSCAN según silhouette:")
        print(valid_db.loc[best_idx])
    else:
        best_eps = eps_values[0]
        best_ms = min_samples_values[0]
        print("\n[WARN] No se encontraron configuraciones DBSCAN válidas (>=2 clusters). Usando fallback.")

    labels_db, db_model, db_metrics = run_dbscan_best_params(X_scaled, best_eps, best_ms)
    print("\n[RESULTADOS] DBSCAN:")
    print(db_metrics)

    # 4) Visualizaciones PCA y t-SNE
    labels_dict = {
        f"KMeans_k={best_k}": labels_km,
        f"Hierarchical_k={best_k}": labels_h,
        f"DBSCAN_eps={best_eps}_ms={best_ms}": labels_db
    }

    print("\n[INFO] Generando visualizaciones PCA y t-SNE...")
    plot_clusters_pca_tsne(X_scaled, labels_dict, title_prefix="Mall Customers")

    # 5) Resumen de métricas
    summary = pd.DataFrame([
        {
            "Modelo": f"K-Means (k={best_k})",
            "Clusters_útiles": len(np.unique(labels_km)),
            "Ruido": 0,
            "Silhouette": km_metrics["silhouette"],
            "Davies-Bouldin": km_metrics["davies_bouldin"],
        },
        {
            "Modelo": f"Jerárquico (k={best_k})",
            "Clusters_útiles": len(np.unique(labels_h)),
            "Ruido": 0,
            "Silhouette": h_metrics["silhouette"],
            "Davies-Bouldin": h_metrics["davies_bouldin"],
        },
        {
            "Modelo": f"DBSCAN (eps={best_eps}, ms={best_ms})",
            "Clusters_útiles": db_metrics["n_clusters"],
            "Ruido": db_metrics["n_noise"],
            "Silhouette": db_metrics["silhouette"],
            "Davies-Bouldin": db_metrics["davies_bouldin"],
        },
    ])

    print("\n[RESUMEN COMPARATIVO]")
    print(summary)


# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    main()
