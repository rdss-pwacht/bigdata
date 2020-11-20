from data_provider import load_train

#Aufgabe1 Flatten category_name
def categoryFlatten(train_data):
    train_data[['cat0', 'cat1', 'cat2','cat3','cat4']] = train_data.category_name.str.split("/",expand=True)
    return train_data
    
#Aufgabe2 Count how often which top level category (cat_0) is present?
def countTopCategory(train_data):
    train_data = categoryFlatten(train_data)
    print("Missing values for each column " ,train_data.isna().sum())
    return train_data['cat0'].value_counts()

train_data = categoryFlatten(load_train())
column_1 = train_data["cat0"]
column_2 = train_data["cat1"]
#print(countTopCategory())
