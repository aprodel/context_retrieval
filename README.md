# context_retrieval


L'outil de context retrieval est accessible via le script "context_finder.py".
L'approche retenue est une recherche de contexte par similarité cosinus calculée sur la base du score TF-IDF de la question et des contextes du dataset.
Le programme est utilisable via la ligne de commande. Ci-après un exemple pour utilisation sur le dataset d'entraînement de BoolQ :

python context_finder.py preprocess "./BoolQ_dataset/train.jsonl"
python context_finder.py compute_TFIDF
python context_finder.py find_context "are persian and farsi the same language" 1

Des informations supplémentaires concernant l'utilisation du script sont accessibles via la commande "python context_finder.py nom_de_la_fonction --help".

