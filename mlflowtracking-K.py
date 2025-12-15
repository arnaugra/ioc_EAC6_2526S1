""" @ IOC - Joan Quintana - 2024 - CE IABD """

import sys
import logging
import shutil
import mlflow

from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

if __name__ == "__main__":

    # TODO
    Ks = [2, 3, 4, 5, 6, 7, 8]

    for k in Ks:
        df = load_dataset('./data/ciclistes.csv')
        df = clean(df)
        true_labels = extract_true_labels(df)
        df = df.drop(columns = ["name"])

        dataset = mlflow.data.from_pandas(df)
        mlflow.start_run(run_name=f"K={k}")
        mlflow.log_input(dataset, context="training")

        model = clustering_kmeans(df, k)
        data_labels = model.labels_

        h_score = round(homogeneity_score(true_labels, data_labels), 5)
        c_score = round(completeness_score(true_labels, data_labels), 5)
        v_score = round(v_measure_score(true_labels, data_labels), 5)

        logging.info('K: %d', k)
        logging.info('H-measure: %.5f', h_score)
        logging.info('C-measure: %.5f', c_score)
        logging.info('V-measure: %.5f', v_score)

        tags = {
            "engineering": "granados_arnau-IOC",
            "release.candidate": "RC1",
            "release.version": "1.1.2",
        }

        mlflow.set_tags(tags)

        mlflow.log_param("K", k)

        mlflow.log_metric("homogeneity", h_score)
        mlflow.log_metric("completeness", c_score)
        mlflow.log_metric("v_measure", v_score)

        mlflow.log_artifact("./data/ciclistes.csv")
        mlflow.end_run()

    print('s\'han generat els runs')
