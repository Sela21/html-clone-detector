from bs4 import BeautifulSoup
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from scipy.sparse import hstack

base_data_folder = 'data'

for tier_folder in sorted(os.listdir(base_data_folder)):
    full_folder_path = os.path.join(base_data_folder, tier_folder)
    if not os.path.isdir(full_folder_path):
        continue

    print(f"\n### Clustering pentru folderul: {tier_folder} ###")

    texts, doms, classes, filenames = [], [], [], []

    for filename in os.listdir(full_folder_path):
        if filename.endswith('.html'):
            file_path = os.path.join(full_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
                # Text
                texts.append(soup.get_text(separator=' ', strip=True))
                
                # DOM (taguri)
                doms.append(' '.join([tag.name for tag in soup.find_all(True)]))

                # Clase CSS
                class_list = []
                for tag in soup.find_all(True):
                    class_attr = tag.get('class')
                    if class_attr:
                        class_list.extend(class_attr)
                classes.append(' '.join(class_list))

                filenames.append(filename)

    if not texts:
        print("⚠️ Nu există fișiere HTML valide în acest folder.")
        continue

    # Vectorizare
    X_text = TfidfVectorizer().fit_transform(texts)
    X_dom = CountVectorizer().fit_transform(doms)
    X_class = CountVectorizer().fit_transform(classes)

    # Combinare (Text + DOM + CSS)
    X_combined = hstack([X_text, X_dom, X_class])

    # Similaritate combinată
    similarity_matrix = cosine_similarity(X_combined)

    # Ajustarea matricei
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0

    # Ajustare clustering DBSCAN - eps ajustat pentru rezultate optime
    clustering = DBSCAN(eps=0.25, min_samples=2, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    # Rezultate clare
    df_results = pd.DataFrame({'Fisier': filenames, 'Cluster': labels})
    df_results_sorted = df_results.sort_values(by='Cluster')

    print(df_results_sorted)
    # Salvare în CSV
    output_csv = f"results_{tier_folder}.csv"
    df_results_sorted.to_csv(output_csv, index=False)
    print(f"✅ Rezultatele pentru {tier_folder} salvate în {output_csv}")
