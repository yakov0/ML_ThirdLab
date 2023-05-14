import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


all_data = pd.read_csv('dataset_group.csv',header=None)

unique_id = list(set(all_data[1]))
#print(len(unique_id)) #Выведем количество id

items = list(set(all_data[2]))
#print(len(items)) #Выведем количество товаров

dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]

#Подготовка данных
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
#print(df)#матрица в 1 столбце id покупателя, далее по столбцам 0 или 1 в зависимости от товара



#Ассоциативный анализ с использованием алгоритма Apriori

# алгоритм apriori с минимальным уровнем поддержки 0.3
#results = apriori(df, min_support=0.3, use_colnames=True)
#results['length'] = results['itemsets'].apply(lambda x: len(x)) #добавление размера набора
#print(results)

#алгоритм apriori с тем же уровнем поддержки, но ограничим максимальный
#размер набора единицей
#results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
#print(results)

#алгоритм apriori и выведем только те наборы, которые имеют размер 2, а также
#количество таких наборов
# results = apriori(df, min_support=0.3, use_colnames=True)
# results['length'] = results['itemsets'].apply(lambda x: len(x))
# results = results[results['length'] == 2]
#print(results)
#print('\nCount of result itemstes = ',len(results))

