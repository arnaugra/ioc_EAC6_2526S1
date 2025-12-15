"""
@ IOC - CE IABD
"""
import os
import logging
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

logging.basicConfig(format='%(message)s', level=logging.INFO)

def load_dataset(path):
    """
    Carrega el dataset de registres dels ciclistes

    arguments:
        path -- dataset

    Returns: dataframe
    """
    return pd.read_csv(path, delimiter=",")

def eda(df_):
    """
    Exploratory Data Analysis del dataframe

    arguments:
        df -- dataframe

    Returns: None
    """
    print("\ninfo: ")
    df_.info()

    print("\nciclistes a cada perfil: ")
    print(df_["name"].value_counts())

    print("\nmitja temps per cada perfil: ")
    print(df_.groupby("name")["tt"].mean())

def clean(df_):
    """
    Elimina les columnes que no són necessàries per a l'anàlisi dels clústers

    arguments:
        df -- dataframe

    Returns: dataframe
    """
    return df_.drop(columns = ["id", "tt"])

def extract_true_labels(df_):
    """
    Guardem les etiquetes dels ciclistes (BEBB, ...)

    arguments:
        df -- dataframe

    Returns: numpy ndarray (true labels)
    """
    return df_["name"]

def visualitzar_pairplot(df_):
    """
    Genera una imatge combinant entre sí tots els parells d'atributs.
    Serveix per apreciar si es podran trobar clústers.

    arguments:
        df -- dataframe

    Returns: None
    """
    os.makedirs("img", exist_ok=True)

    # Seleccionem només les columnes numèriques que ens interessen
    cols = ["tp", "tb", "cluster"]
    # Pairplot amb els perfils com a hue
    imatge = sns.pairplot(df_[cols], hue="cluster", diag_kind="kde", corner=False)
    imatge.fig.suptitle("Pairplot dels temps per perfil", y=1.02)

    # guardar imatge generada (?)
    imatge.fig.savefig("img/pairplot.png", dpi=300, bbox_inches='tight')

    plt.show()

def clustering_kmeans(data, n_clusters=4):
    """
    Crea el model KMeans de sk-learn, amb 4 clusters (estem cercant 4 agrupacions)
    Entrena el model

    arguments:
        data -- les dades: tp i tb

    Returns: model (objecte KMeans)
    """
    model = KMeans(n_clusters = n_clusters, random_state = 42)
    return model.fit(data)

def visualitzar_clusters(data, labels):
    """
    Visualitza els clusters en diferents colors. Provem diferents combinacions de parells d'atributs

    arguments:
        data -- el dataset sobre el qual hem entrenat
        labels -- l'array d'etiquetes a què pertanyen les dades
                  (hem assignat les dades a un dels 4 clústers)

    Returns: None
    """
    os.makedirs("img", exist_ok=True)

    data["cluster"] = labels

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.scatterplot(
        data=data,
        x="tp",
        y="tb",
        hue="cluster",
        ax=axes[0]
    )
    axes[0].set_title("tp vs tb")
    axes[0].set_xlabel("tp")
    axes[0].set_ylabel("tb")

    sns.boxplot(
        data=data,
        x="cluster",
        y="tp",
        ax=axes[1]
    )
    axes[1].set_title("tp")
    axes[1].set_xlabel("cluster")
    axes[1].set_ylabel("tp")

    sns.boxplot(
        data=data,
        x="cluster",
        y="tb",
        ax=axes[2]
    )
    axes[2].set_title("tb")
    axes[2].set_xlabel("cluster")
    axes[2].set_ylabel("tb")

    fig.savefig("img/clusters.png", dpi=300, bbox_inches='tight')
    plt.show()

def associar_clusters_patrons(tipus_, model):
    """
    Associa els clústers (labels 0, 1, 2, 3) als patrons de comportament (BEBB, BEMB, MEBB, MEMB).
    S'han trobat 4 clústers però aquesta associació encara no s'ha fet.

    arguments:
    tipus -- un array de tipus de patrons que volem actualitzar associant els labels
    model -- model KMeans entrenat

    Returns: array de diccionaris amb l'assignació dels tipus als labels
    """
    # proposta de solució

    dicc = {'tp':0, 'tb': 1}

    logging.info('Centres:')
    for j in range(len(tipus_)):
        tp = model.cluster_centers_[j][dicc['tp']]
        tb = model.cluster_centers_[j][dicc['tb']]
        logging.info('%s: (tp: %s tb: %s)', j, tp, tb)

    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    for j, center in enumerate(model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus_[0].update({'label': ind_label_0})
    tipus_[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if model.cluster_centers_[lst[0]][0] < model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus_[1].update({'label': ind_label_1})
    tipus_[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus_)
    return tipus_

def generar_informes(df_, tipus_):
    """
    Generació dels informes a la carpeta informes/.
	Tenim un dataset de ciclistes i 4 clústers, i generem
    4 fitxers de ciclistes per cadascun dels clústers

    arguments:
        df -- dataframe
        tipus -- objecte que associa els patrons de comportament amb els labels dels clústers

    Returns: None
    """
    os.makedirs("informes", exist_ok=True)

    for tip in tipus_:
        fitxer_ = f"{tip['name']}.txt"
        df_tipus = df_[df_['cluster'] == tip['label']]
        # with open(f'informes/{fitxer}', 'w') as f:
        #     f.write(df_tipus)
        df_tipus.to_csv(f'informes/{fitxer_}', index = False)

    logging.info('S\'han generat els informes en la carpeta informes/\n')

def nova_prediccio(dades, model):
    """
    Passem nous valors de ciclistes, per tal d'assignar aquests valors a un dels 4 clústers

    arguments:
        dades -- llista de llistes, que segueix l'estructura 'id', 'tp', 'tb', 'tt'
        model -- clustering model
    Returns: (dades agrupades, prediccions del model)
    """
    return model.predict(dades[["tp", "tb"]])

# ----------------------------------------------

if __name__ == "__main__":

    PATH_DATASET = './data/ciclistes.csv'
    """
    TODO:
    load_dataset
    EDA
    clean
    extract_true_labels
    eliminem el tipus, ja no interessa .drop('tipus', axis=1)
    visualitzar_pairplot
    clustering_kmeans
    pickle.dump(...) guardar el model
    mostrar scores i guardar scores
    visualitzar_clusters
    """

    # càrrega dataset
    df = load_dataset(PATH_DATASET)

    # EDA
    eda(df)

    # neteja dades
    df = clean(df)

    # extracció de labels
    labels_originals = extract_true_labels(df)
    df = df.drop(columns = ["name"])

    # clustering amb KMeans
    clustering_model = clustering_kmeans(df)
    clusters_predits = clustering_model.labels_

    # guardar model
    os.makedirs("model", exist_ok=True)
    with open("model/clustering_model.pkl", "wb") as fitxer:
        pickle.dump(clustering_model, fitxer)

    # guaardar homogeneitat, completesa i v-measure
    scores = {
        "homogeneity": homogeneity_score(labels_originals, clusters_predits),
        "completeness": completeness_score(labels_originals, clusters_predits),
        "v_measure": v_measure_score(labels_originals, clusters_predits)
    }
    print(scores)
    with open("model/scores.pkl", "wb") as fitxer:
        pickle.dump(scores, fitxer)

    # visualitzar clusters
    df_clusters = df.copy()
    df_clusters['cluster'] = clusters_predits
    #afegir labels_originals
    df_clusters['label'] = labels_originals

    visualitzar_pairplot(df_clusters)
    visualitzar_clusters(df, clusters_predits)


    # array de diccionaris que assignarà els tipus als labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

    """
    afegim la columna label al dataframe
    associar_clusters_patrons(tipus, clustering_model)
    guardem la variable tipus a model/tipus_dict.pkl
    generar_informes
    """

    tipus_associats = associar_clusters_patrons(tipus, clustering_model)

    with open("model/tipus_dict.pkl", "wb") as fitxer:
        pickle.dump(tipus_associats, fitxer)

    # generar informes

    generar_informes(df_clusters, tipus_associats)


    # Classificació de nous valors
    nous_ciclistes = [
        [500, 3230, 1430, 4670], # BEBB
        [501, 3300, 2120, 5420], # BEMB
        [502, 4010, 1510, 5520], # MEBB
        [503, 4350, 2200, 6550] # MEMB
    ]

    """
    nova_prediccio

    #Assignació dels nous valors als tipus
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('tipus %s (%s) - classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
    """
    df_nous_ciclistes = pd.DataFrame(
        nous_ciclistes,
        columns=["id", "tp", "tb", "tt"]
    )

    pred = nova_prediccio(df_nous_ciclistes, clustering_model)

    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        print(f'tipus {df_nous_ciclistes.index[i]} ({t[0]['name']}) - classe {p}')
