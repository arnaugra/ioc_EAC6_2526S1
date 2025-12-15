"""
Modul per generar datasets de temps de recorregut de ciclistes
"""
import os
import logging
import random
import numpy as np

def generar_dataset(num, ind, diccionari):
    """
    Genera els temps dels ciclistes, de forma aleatòria, però en base a la informació del diccionari
    TODO: completar arguments, return. num és el número de files/ciclistes
	a generar. ind és l'index/identificador/dorsal.
    """
    ciclistes = []

    for j in range(num):
        tipus_ciclista = random.choice(diccionari)
        name = tipus_ciclista["name"]
        temps_pujada = np.random.normal(tipus_ciclista["mu_p"], tipus_ciclista["sigma"])
        temps_baixada = np.random.normal(tipus_ciclista["mu_b"], tipus_ciclista["sigma"])
        temps_total = temps_pujada + temps_baixada

        ciclista = {
            "id": ind + j,
            "name": name,
            "tp": round(temps_pujada, 3),
            "tb": round(temps_baixada, 3),
            "tt": round(temps_total, 3)
        }

        ciclistes.append(ciclista)

    return ciclistes

if __name__ == "__main__":

    STR_CICLISTES = 'data/ciclistes.csv'


    # BEBB: bons escaladors, bons baixadors
    # BEMB: bons escaladors, mal baixadors
    # MEBB: mal escaladors, bons baixadors
    # MEMB: mal escaladors, mal baixadors

    # Port del Cantó (18 Km de pujada, 18 Km de baixada)
    # pujar a 20 Km/h són 54 min = 3240 seg
    # pujar a 14 Km/h són 77 min = 4268 seg
    # baixar a 45 Km/h són 24 min = 1440 seg
    # baixar a 30 Km/h són 36 min = 2160 seg
    MU_P_BE = 3240 # mitjana temps pujada bons escaladors
    MU_P_ME = 4268 # mitjana temps pujada mals escaladors
    MU_B_BB = 1440 # mitjana temps baixada bons baixadors
    MU_B_MB = 2160 # mitjana temps baixada mals baixadors
    SIGMA = 240 # 240 s = 4 min

    dicc = [
        {"name":"BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name":"MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name":"MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    NUMERO_CICLISTES = 250 # ?
    dades = generar_dataset(NUMERO_CICLISTES, 0, dicc)

    os.makedirs("data", exist_ok=True)

    with open(STR_CICLISTES, "w", encoding="utf-8") as fitxer:
        for i, cic in enumerate(dades):
            if i == 0:
                fitxer.write("id,name,tp,tb,tt\n")
            fitxer.write(f'{cic["id"]},{cic["name"]},{cic["tp"]},{cic["tb"]},{cic["tt"]}\n')

    logging.info("s'ha generat data/ciclistes.csv")
