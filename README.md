# portcanto
__portcanto__ es un projecte de simulació de mitjes de temps en pujades i baixades d'un port de muntana (port del cantó)

Es creen dades sintetiques per poder fer un analisis de les dades amb IA (Clustering)

Consta de tres parts:
 - generardataset: genera a la carpeta `data/` el dataset amb ades generades amb 4 comportaments (4 clusters)
 - clusterciclistes: fa l'analisi del dataset i l'entrenament del model (amb 4 clusters) i fa una predicció amb dades noves
 - mlflowtracking-K: fa el proces del `clusterciclistes` pero cambiant el nombre de clusters desde 2 fins a 8 clusters per comprobar els diferents resultats

# Execució
Creear l'entorn virtual
```
$ python -m venv venv

$ . venv/Scripts/activate
```

Intal·lar les dependencies
```
$ pip install -r requirements.txt
```

Executar scripts
```
# generardades
python generardataset.py
```
```
# clustersciclistes
python clustersciclistes.py
```
```
# mlflowtracking-K
python mlflowtracking-K.py
```

# Testing
Des de l'arrel del projecte:
```
$ python -m unittest discover -s tests
```
