from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from datetime import datetime

# Funcție de vizualizare și salvare grafic
def vizualizeaza_clustering(X_combined, labels, tier_folder, timestamp):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_combined.toarray())

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=40, alpha=0.7)
    plt.title(f'Clustering - {tier_folder}')
    plt.xlabel('Componenta Principală 1')
    plt.ylabel('Componenta Principală 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.tight_layout()

    diagram_folder = os.path.join("diagrams", f"diagrams_{tier_folder}")
    os.makedirs(diagram_folder, exist_ok=True)
    diagram_path = os.path.join(diagram_folder, f"clustering_{tier_folder}_{timestamp}.png")
    plt.savefig(diagram_path)
    plt.close()

    print(f" Diagramă salvată: {diagram_path}")

# Directorul cu fișierele HTML
base_data_folder = 'data'

for tier_folder in sorted(os.listdir(base_data_folder)):
    full_folder_path = os.path.join(base_data_folder, tier_folder)
    if not os.path.isdir(full_folder_path):
        continue

    try:
        print(f"\n Procesare folder: {tier_folder}")

        texts, doms, classes, filenames = [], [], [], []

        for filename in os.listdir(full_folder_path):
            if filename.endswith('.html'):
                file_path = os.path.join(full_folder_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                    texts.append(soup.get_text(separator=' ', strip=True))
                    doms.append(' '.join([tag.name for tag in soup.find_all(True)]))

                    class_list = []
                    for tag in soup.find_all(True):
                        class_attr = tag.get('class')
                        if class_attr:
                            class_list.extend(class_attr)
                    classes.append(' '.join(class_list))
                    filenames.append(filename)

        if not texts:
            print(" Nu s-au găsit fișiere HTML valide.")
            continue

        X_text = TfidfVectorizer().fit_transform(texts)
        X_dom = CountVectorizer().fit_transform(doms)
        X_class = CountVectorizer().fit_transform(classes)
        X_combined = hstack([X_text, X_dom, X_class])

        similarity_matrix = cosine_similarity(X_combined)
        distance_matrix = 1 - similarity_matrix
        distance_matrix[distance_matrix < 0] = 0

        clustering = DBSCAN(eps=0.25, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        df_results = pd.DataFrame({'Fisier': filenames, 'Cluster': labels})
        df_results_sorted = df_results.sort_values(by='Cluster')
        print(df_results_sorted)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = os.path.join("results", f"results_{tier_folder}")
        os.makedirs(results_folder, exist_ok=True)
        output_csv = os.path.join(results_folder, f"results_{tier_folder}_{timestamp}.csv")
        df_results_sorted.to_csv(output_csv, index=False)

        print(f" Rezultate CSV salvate: {output_csv}")

        vizualizeaza_clustering(X_combined, labels, tier_folder, timestamp)

    except Exception as e:
        print(f" Eroare la {tier_folder}: {e}")
