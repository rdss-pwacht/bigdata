# from categoryProvider import categoryFlatten
import logging
import re

from pathlib import Path
from typing import Iterable
from typing import Set

import altair as alt
import nltk
import pandas as pd

from google.cloud import bigquery
from nltk.corpus import stopwords
from scipy import stats

import mercari.cache as cache


logging.basicConfig(
    level=logging.INFO,
    style="{",
    format="{asctime}.{msecs:03.0f} {levelname:>7s} {process:d} "
    + "--- [{threadName:>15.15}] {name:<40.40s} : {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

project = "rd-rdss-playground"
table_id = "mercari_price.train"


# Aufgabe 4 Find the most commonly used words in the item_description column.
def compute_word_counts(train_data: pd.DataFrame) -> pd.DataFrame:
    return (
        train_data.item_description_without_stopwords.str.split(expand=True)
        .stack()
        .value_counts()
        .rename_axis("word")
        .reset_index(name="count")
    )


# find common words
def get_word_counts_gt_threshold(
    word_counts: pd.DataFrame, threshold: int = 200
) -> pd.DataFrame:
    return word_counts.loc[word_counts["count"] > threshold].reset_index(drop=True)


# find uncommon words
def get_word_counts_le_threshold(
    word_counts: pd.DataFrame, threshold: int = 200
) -> pd.DataFrame:
    return word_counts.loc[word_counts["count"] <= threshold].reset_index(drop=True)


# download once and store local the corpora stopwords
def use_stopwords_local() -> Set[str]:
    path = str(Path().absolute())
    if Path(path + "/corpora/stopwords/").is_dir():
        nltk.data.path.append(path)
        return set(stopwords.words("english"))

    nltk.download("stopwords", download_dir=path)
    nltk.data.path.append(path)
    return set(stopwords.words("english"))


# Aufgabe 5 Remove words from the item_description that carry no information
# e.g. "the", "but", "a" "is" etc.
def remove_stopwords_from_item_description(train_data: pd.DataFrame) -> pd.DataFrame:
    stop = use_stopwords_local()
    train_data = train_data[train_data["item_description"].notnull()]
    train_data["item_description"] = train_data["item_description"].apply(
        lambda s: re.sub(r"[^A-Za-z0-9 ]+", "", s.casefold())
    )
    train_data["item_description_without_stopwords"] = train_data[
        "item_description"
    ].apply(lambda x: " ".join([word for word in x.split() if word not in (stop)]))
    return train_data


# Aufgabe 6
def fuzzy_search_category_in_desc(train_data: pd.DataFrame) -> None:
    flattenCategories = {"cat0", "cat1", "cat2", "cat3", "cat4"}
    for categoryColumn in flattenCategories:
        categoryNames = train_data[categoryColumn].dropna().unique()
        newDataWithCategoryNameCorr = []
        for categoryName in categoryNames:
            categoryTotal = (train_data[categoryColumn].values == categoryName).sum()
            filtered = train_data[
                train_data["item_description_without_stopwords"].str.contains(
                    categoryName, na=False
                )
            ].shape[0]
            corr = (filtered / categoryTotal) * 100
            if corr > 0:
                newDataWithCategoryNameCorr.append([categoryName, corr])

        data = pd.DataFrame(newDataWithCategoryNameCorr, columns=["category", "corr"])

        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("category", axis=alt.Axis(title="Category Name")),
                y=alt.Y(
                    "corr",
                    axis=alt.Axis(title="Category name and description correlation"),
                ),
            )
        )
        chart.save(categoryColumn + "_correlation_with_description.html")


def browse_table_data(project, table_id) -> pd.DataFrame:
    client = bigquery.Client(project)
    rows_iter = client.list_rows(table_id)
    dataframe = rows_iter.to_dataframe()
    return dataframe


def drop_uncommon_words_from_desc(
    data: pd.DataFrame, uncommon_words: Iterable[str]
) -> pd.DataFrame:
    return data.assign(
        item_description_common_words=data["item_description_without_stopwords"].apply(
            lambda x: " ".join(
                [word for word in x.split() if word not in (uncommon_words)]
            )
        )
    )


def main() -> int:
    logger.info("Starting application...")
    cache_cfg = cache.DefaultCacheConfig(enabled=True, directory="./cache_dir")
    threshold_common: int = 200

    logger.info("Loading source data...")
    source_df = cache.cache_dataframe(table_id, browse_table_data, project, table_id)(
        cache_cfg
    )
    logger.info("Done loading source data")

    logger.info("Removing stopwords...")
    train_data = cache.cache_dataframe(
        "without_stopwords", remove_stopwords_from_item_description, source_df
    )(cache_cfg)
    logger.info("Removed stopwords")

    logger.info("Computing word counts...")
    word_counts = cache.cache_dataframe("word_counts", compute_word_counts, train_data)(
        cache_cfg
    )
    logger.info("Computed word counts")

    logger.info("Finding uncommon words...")
    uncommonly_used = cache.cache_dataframe(
        "uncommon_words", get_word_counts_le_threshold, word_counts, threshold_common
    )(cache_cfg)
    logger.info("Found uncommon words")

    logger.info("Drop uncommon words from item description...")
    train_data = cache.cache_dataframe(
        "train_data_common",
        drop_uncommon_words_from_desc,
        train_data,
        uncommonly_used["word"],
    )(cache_cfg)
    logger.info("Dropped uncommon words from item description")

    logger.info("Splitting words...")
    train_data["common_words"] = train_data["item_description_common_words"].str.split()
    logger.info("Done splitting words")

    logger.info("Explode words...")
    small_df = train_data[["price", "common_words"]]
    not_so_small_df = small_df.explode("common_words")
    logger.info("Exploded words")

    logger.info("Grouping...")
    # get price dataframes per word
    grpd = [x["price"] for _, x in not_so_small_df.groupby("common_words")]
    logger.info("Done grouping")

    # Pseudocode. Google how to do this.
    # not_so_small_df.groupby("common_words")\
    #    .sort_by("price", desc=True) \
    #    .take_top(10)

    # is there a correlation between price and word?
    F, p = stats.f_oneway(*grpd)
    print(F)
    print(p)

    # average price per word
    avg_prices_by_word = not_so_small_df.groupby("common_words").agg(
        avg_price=("price", "mean")
    )
    print(avg_prices_by_word)

    # logger.info("Generate dummies...")
    # my_dummies = pd.get_dummies(not_so_small_df, prefix="", prefix_sep="")
    # logger.info("Generated dummies")
    # print(my_dummies.tail())

    # fuzzy_search_category_in_desc(train_data)
    logger.info("Application terminated successfully")
    return 0


if __name__ == "__main__":
    main()
