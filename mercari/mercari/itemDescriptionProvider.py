from data_provider import load_train
from categoryProvider import categoryFlatten
from collections import Counter
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import altair as alt
import pandas as pd
import time
import re

#Aufgabe 4 Find the most commonly used words in the item_description column.
def getMostCommonUsedWords(train_data):
    train_data = train_data.assign(item_description=train_data.item_description.apply(lambda s:re.sub(r'[^A-Za-z0-9 ]+', '', s.casefold())))
    commonlyused:pd.Series = train_data.item_description.str.split(expand=True).stack().value_counts()
    commonlyused.to_csv('commonlyused.csv')

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
    train_data['item_description_without_stopwords'] = train_data['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #print(train_data[['item_description','item_description_without_stopwords']].tail())
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


def main():
    t0 = time.time()
    train_data = categoryFlatten(load_train())
    train_data = removeStopWordsFromItemDescription(train_data)
    #fuzzySearchCategoryInDescription(train_data)
    getMostCommonUsedWords(train_data)
    print(f'Done in {time.time() - t0:.0f} s')

if __name__ == '__main__':
    main()