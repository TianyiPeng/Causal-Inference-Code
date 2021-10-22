import numpy as np
import pandas as pd
import pickle

def read_data(scenario='beer'):
    if scenario == 'beer':
        df = pd.read_csv(open('dataset/beer_filter.csv'))
        beer_data = np.array(df.drop(['ID'], axis = 1))
        return beer_data

    if scenario == 'tobacco':
        df = pd.read_csv('dataset/prop99.csv')  ##input csv file
        df = df[df['SubMeasureDesc'] == 'Cigarette Consumption (Pack Sales Per Capita)'] ## extract the metric that we want
        pivot = df.pivot_table(values='Data_Value', index='LocationDesc', columns=['Year']) ## obtain the desired pivot table: index: state-name, column: year, value: per captita consumption

        dfProp99 = pd.DataFrame(pivot.to_records())
        allColumns = dfProp99.columns.values
        states = list(np.unique(dfProp99['LocationDesc']))
        years = allColumns[1:]
        O = dfProp99[years].values
        select = []
        remove_list = ['Massachusetts', 'Arizona', 'Oregon', 'Florida', 'Alaska', 'Hawaii', 'Maryland', 'Michigan', 'New Jersey', 'New York', 'Washington', 'District of Columbia', 'California']
        for i in range(O.shape[0]):
            if (states[i] not in remove_list):
                select.append(i)
        O = O[select, :]

        end_index = 2001 - 1970
        return O[:, :end_index]

    if scenario == 'Covid':
        O = pickle.load(open('dataset/covid.p', 'rb'))
        O = pickle.load(open('dataset/covid_2.p', 'rb'))

        O = pickle.load(open('dataset/covid_3.p', 'rb')) ##unnormalized infection rate
        return O

    if scenario == 'sales':
        O = pickle.load(open('dataset/sales.p', 'rb'))
        return O