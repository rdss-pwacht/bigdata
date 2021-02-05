# from categoryProvider import categoryFlatten
import nltk
import logging
from nltk.corpus import stopwords
from pathlib import Path
import altair as alt
import pandas as pd
import re
from typing import NamedTuple
from typing import Any
from typing import Callable
from typing import Protocol
import os
from google.cloud import bigquery

logger: logging.Logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

project = "rd-rdss-playground"
table_id = "mercari_price.train"

# Aufgabe 4 Find the most commonly used words in the item_description column.


def getMostCommonUsedWords(train_data):
    # train_data = train_data.assign(item_description=train_data.item_description_without_stopwords.apply(lambda s:re.sub(r'[^A-Za-z0-9 ]+', '', s.casefold())))
    df = (
        train_data.item_description_without_stopwords.str.split(expand=True)
        .stack()
        .value_counts()
        .rename_axis("word")
        .reset_index(name="count")
    )

    # df = commonlyused.to_frame()
    # print(df.tail(200))
    # df.columns = ['word', 'count']
    df.to_csv("commonlyused.csv", index=False)


# Cleanup-CommonUsedWords
def cleanUpCommonUsedWords():
    common_used_words_data: pd.DataFrame = pd.read_csv("commonlyused.csv")
    common_used_words_data = common_used_words_data.loc[
        common_used_words_data["count"] > 200
    ]
    common_words: pd.DataFrame = common_used_words_data
    return common_words
    # common_used_words_data.to_csv('commonlyusedcleaned.csv', index=False)


# Cleanup-CommonUsedWords
def getUncommonUsedWords() -> pd.DataFrame:
    logger.info("start uncommon words")
    common_used_words_data: pd.DataFrame = pd.read_csv("commonlyused.csv").astype(
        {"word": str, "count": int}
    )
    uncommon_used_words_data = common_used_words_data.loc[
        common_used_words_data["count"] < 200
    ]
    return uncommon_used_words_data


# download once and store local the corpora stopwords
def useStopWordsLocal():
    path = str(Path().absolute())
    if Path(path + "/corpora/stopwords/").is_dir():
        nltk.data.path.append(path)
        return set(stopwords.words("english"))

    nltk.download("stopwords", download_dir=path)
    nltk.data.path.append(path)
    return set(stopwords.words("english"))


# Aufgabe 5 Remove words from the item_description that carry no information, e.g. "the", "but", "a" "is" etc.
def removeStopWordsFromItemDescription(train_data):
    stop = useStopWordsLocal()
    train_data = train_data[train_data["item_description"].notnull()]
    train_data = train_data.assign(
        item_description=train_data.item_description.apply(
            lambda s: re.sub(r"[^A-Za-z0-9 ]+", "", s.casefold())
        )
    )
    train_data["item_description_without_stopwords"] = train_data[
        "item_description"
    ].apply(lambda x: " ".join([word for word in x.split() if word not in (stop)]))
    return train_data


# Aufgabe 6
def fuzzySearchCategoryInDescription(train_data):
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


def browse_table_data(project, table_id):
    client = bigquery.Client(project)
    rows_iter = client.list_rows(table_id)
    dataframe = rows_iter.to_dataframe()
    return dataframe


class MyCacheConfig(NamedTuple):
    enabled: bool
    directory: str


class _CacheConfig(Protocol):
    enabled: bool
    directory: str


def cache_dataframe(
    cache_id: str,
    cache_miss_callback: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
) -> Callable[[_CacheConfig], pd.DataFrame]:
    def factory(cache_config: _CacheConfig) -> pd.DataFrame:
        if cache_config.enabled:
            cache_dir = cache_config.directory
            os.makedirs(cache_dir, exist_ok=True)
            file_path = f"{cache_dir}/{cache_id}.pkl.bz2"
            if os.path.exists(file_path):
                result = pd.read_pickle(file_path)
            else:
                result = cache_miss_callback(*args, **kwargs)
                result.to_pickle(file_path)
        else:
            result = cache_miss_callback(*args, **kwargs)
        return result

    return factory


def browse_cache_data() -> pd.DataFrame:
    logger.info("Loading data...")
    my_cache_conf = MyCacheConfig(True, "./cache_dir")
    result = cache_dataframe(table_id, browse_table_data, project, table_id)(
        my_cache_conf
    )
    logger.info("Done loading data")
    return result


def main():
    logger.info("Starting application...")
    # train_data = categoryFlatten(browse_cache_data())
    #train_data = removeStopWordsFromItemDescription(browse_cache_data())
    #getMostCommonUsedWords(train_data)
    #uncommonly_used = getUncommonUsedWords()

    # for word in tqdm(uncommonly_used['word']):
    # train_data['item_description_without_stopwords'] = train_data['item_description_without_stopwords'].str.replace(word, '')

    # train_data['item_description_common_words'] = train_data['item_description_without_stopwords'].str.replace("|".join(uncommonly_used['word']), '')
    #train_data["item_description_common_words"] = train_data[
    #    "item_description_without_stopwords"
    #].apply(
    #   lambda x: " ".join(
    #        [word for word in x.split() if word not in (uncommonly_used["word"])]
    #    )
    #)

    #drop_columns = ["train_id", "name", "item_condition_id", "shipping", "item_description", "item_description_without_stopwords", "brand_name", "category_name"]
    #train_data.drop(columns=drop_columns, axis=0, inplace=True)
    #train_data.to_csv("price2itemdescriptioncommonwords.csv", index=False)

    df = pd.read_csv("price2itemdescriptioncommonwords.csv")
    s = df.item_description_common_words.str.get_dummies(sep=" ")
    print(s.tail())
    # pd.get_dummies(df, columns=['type'])
    # fuzzySearchCategoryInDescription(train_data)
    logger.info("Application terminated successfully")


if __name__ == "__main__":
    main()
