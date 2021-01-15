from categoryProvider import categoryFlatten
from collections import Counter
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import altair as alt
import pandas as pd
import re
from typing import NamedTuple
from time import sleep
from typing import Any
from typing import Callable
from typing import Protocol
import os
import string
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer 
pd.options.mode.chained_assignment = None  # default='warn'


#Aufgabe 4 Find the most commonly used words in the item_description column.
def getMostCommonUsedWords(train_data):
    #string.punctuation
    train_data['name_len'] = train_data['name'].apply(lambda x: len(x))
    train_data['des_len'] = train_data['item_description'].apply(lambda x: len(x))
    train_data['name_desc_len_ratio'] = train_data['name_len']/train_data['des_len']
    train_data['desc_word_count'] = train_data['item_description'].apply(lambda x: len(x.split()))
    train_data['mean_des'] = train_data['item_description'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
    train_data['name_word_count'] = train_data['name'].apply(lambda x: len(x.split()))
    #example = train_data[['name_len', 'des_len', 'name_desc_len_ratio', 'desc_word_count', 'mean_des','name_word_count']]

    #train_data = train_data.assign(item_description=train_data.item_description_without_stopwords.apply(lambda s:re.sub(r'[^A-Za-z0-9 ]+', '', s.casefold())))
    commonlyused:pd.Series = train_data.item_description_without_stopwords.str.split(expand=True).stack().value_counts()
    
    commonlyused.to_csv('commonlyused.csv')

#Cleanup-CommonUsedWords
def cleanUpCommonUsedWords():
   common_used_words_data:pd.DataFrame = pd.read_csv('commonlyused.csv')
   #282807-113610
   common_used_words_data.drop(common_used_words_data.tail(169197).index, inplace=True)
   #print(common_used_words_data)
   common_used_words_data.to_csv('commonlyusedcleaned.csv', index=False)


#download once and store local the corpora stopwords
def useStopWordsLocal():
    path = str(Path().absolute())
    if Path(path + '/corpora/stopwords/').is_dir():
        nltk.data.path.append(path)
        return set(stopwords.words("english"))

    nltk.download('stopwords',download_dir = path)
    nltk.data.path.append(path)
    return set(stopwords.words("english"))

#Aufgabe 5 Remove words from the item_description that carry no information, e.g. "the", "but", "a" "is" etc.
def removeStopWordsFromItemDescription(train_data):
    stop = useStopWordsLocal()
    train_data = train_data[train_data['item_description'].notnull()]
    train_data = train_data.assign(item_description=train_data.item_description.apply(lambda s:re.sub(r'[^A-Za-z0-9 ]+', '', s.casefold())))
    
    train_data['item_description_without_stopwords'] = train_data['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    return train_data

#Aufgabe 6 
def fuzzySearchCategoryInDescription(train_data):
    flattenCategories = {'cat0', 'cat1', 'cat2', 'cat3', 'cat4'}
    for categoryColumn in flattenCategories:
        categoryNames = train_data[categoryColumn].dropna().unique()
        newDataWithCategoryNameCorr= []
        for categoryName in categoryNames:
            categoryTotal = (train_data[categoryColumn].values == categoryName).sum()
            filtered = train_data[train_data['item_description_without_stopwords'].str.contains(categoryName, na=False)].shape[0]
            corr = (filtered / categoryTotal) * 100
            if corr > 0:
                newDataWithCategoryNameCorr.append([categoryName, corr])
        
        data = pd.DataFrame(newDataWithCategoryNameCorr, columns=['category', 'corr'])
    
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('category', axis=alt.Axis(title='Category Name')),
            y=alt.Y('corr', axis=alt.Axis(title='Category name and description correlation'))
            )
        chart.save(categoryColumn + '_correlation_with_description.html')

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

def browse_cache_data():
    my_cache_conf = MyCacheConfig(True, "./cache_dir")
    result = cache_dataframe(table_id, expensive_function, 2, "hello world", c="test")(my_cache_conf)
    return result



def main():
    #train_data = categoryFlatten(browse_cache_data())
    #train_data = removeStopWordsFromItemDescription(browse_cache_data())
    #fuzzySearchCategoryInDescription(train_data)
    #getMostCommonUsedWords(train_data)
    cleanUpCommonUsedWords()
   

if __name__ == '__main__':
    main()




 


