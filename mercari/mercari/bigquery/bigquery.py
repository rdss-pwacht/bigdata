import os
from google.cloud import bigquery
import pandas as pd
from .cache import cache_dataframe
from typing import NamedTuple
from time import sleep

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


def list_datasets():
    client = bigquery.Client()
    datasets = list(client.list_datasets())  # Make an API request.
    project = client.project

    if datasets:
        print("Datasets in project {}:".format(project))
        for dataset in datasets:
            print("\t{}".format(dataset.dataset_id))
    else:
        print("{} project does not contain any datasets.".format(project))
    # [END bigquery_list_datasets]

def browse_table_data(project,table_id):
    client = bigquery.Client(project)
    rows_iter = client.list_rows(table_id)
    dataframe = rows_iter.to_dataframe()
    return dataframe

project = 'rd-rdss-playground'
table_id = 'mercari_price.train'

class MyCacheConfig(NamedTuple):
    enabled: bool
    directory: str
def expensive_function(a: int, b: str, c: str) -> pd.DataFrame:
    return browse_table_data(project, table_id)

my_cache_conf = MyCacheConfig(True, "./cache_dir")
result = cache_dataframe(table_id, expensive_function, 2, "hello world", c="test")(my_cache_conf)
print(result.tail())