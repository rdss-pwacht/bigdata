from pathlib import Path
import pandas as pd


def load_train(filePath="train.tsv"):
    pd.set_option("mode.chained_assignment", None)
    path = str(Path().absolute())
    df_train = pd.read_csv(path + "/" + filePath, sep="\t")
    return df_train
