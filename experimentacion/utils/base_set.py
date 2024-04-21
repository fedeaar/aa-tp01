import pandas as pd
from utils.train_test_split import train_test_split

# we assume we are running from ./experimentacion
gene_dataset = pd.read_csv("../catedra/datos/data.csv", delimiter=',', encoding="utf-8")
X = gene_dataset.drop("target", axis=1).to_numpy()
y = gene_dataset.target.to_numpy()

# we set a seed value to guarantee dev. re-runs do not change the data split (also, to allow reproduction)
seed = 0x2031
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=seed)
