from typing import NamedTuple

import pandas as pd

from data_provider import load_train

from mercari.bigquery.bigquery import browse_table_data
from mercari.bigquery.cache import cache_dataframe


# Aufgabe1 Flatten category_name
def categoryFlatten(train_data):
    train_data[
        ["cat0", "cat1", "cat2", "cat3", "cat4"]
    ] = train_data.category_name.str.split("/", expand=True)
    return train_data


# Aufgabe2 Count how often which top level category (cat_0) is present?
def countTopCategory(train_data):
    train_data = categoryFlatten(train_data)
    print("Missing values for each column ", train_data.isna().sum())
    return train_data["cat0"].value_counts()


project = "rd-rdss-playground"
table_id = "mercari_price.train"


class MyCacheConfig(NamedTuple):
    enabled: bool
    directory: str


def expensive_function(a: int, b: str, c: str) -> pd.DataFrame:
    return browse_table_data(project, table_id)


my_cache_conf = MyCacheConfig(True, "./cache_dir")
result = cache_dataframe(table_id, expensive_function, 2, "hello world", c="test")(
    my_cache_conf
)

train_data = categoryFlatten(result)
column_1 = train_data["cat0"]
column_2 = train_data["cat1"]

print(result.tail())

# print(countTopCategory())
