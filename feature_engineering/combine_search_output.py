
def title():

    print('++++++++++++++++++++')
    print('Script: combine_search_output.py')
    print('Description: concatenates all of the search output, and processes all text attributes into one message')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import pandas as pd
import numpy as np

def get_desc(def_desc, co_desc):

    final_list = []
    for entry in co_desc + def_desc:
        if pd.notnull(entry) and entry not in final_list:
            final_list.append(entry)

    return 'none' if not final_list else '; '.join(final_list)

def get_message(row_data):

    _, _, mer_cat, def_brand, def_desc, co_brand, co_desc, _, _ = row_data

    msg = 'Brand 1: ' + str(co_brand) + '. '
    msg += 'Brand 2: ' + str(def_brand) + '. '
    msg += 'Merchant Category: ' + ('none' if pd.isnull(mer_cat) else str(mer_cat)) + '. '
    msg += 'Description: ' + get_desc(eval(def_desc), eval(co_desc)) + '.'

    return msg

def check_log_norm(amounts, ep=1):

    if -ep <= amounts.mean() <= ep:
        print('Amounts already normalized')
        return amounts

    print('Log-normalizing amounts')
    log_amounts = np.log(amounts)
    return (log_amounts - log_amounts.mean()) / log_amounts.std()

def combine_outputs(dataset):

    index_pairs = (['0-10000'] if dataset == 'testing'
                        else ['0-10000', '10000-20000', '20000-30000', '30000-40000'])
    dataset_files = ['../data/search_output/' + dataset + '_reduced_cleaned_csv-rows-' + pair + '.csv' for pair in index_pairs]

    df = pd.concat([pd.read_csv(d_file) for d_file in dataset_files], ignore_index=True, sort=False)

    messages = []

    for i in range(df.values.shape[0]):

        if i % 100 == 0:
            print(i, 'of', df.values.shape[0] - 1, 'messages processed')

        messages.append(get_message(df.values[i,:]))

    df_final = pd.DataFrame({
        'Amount' : check_log_norm(df['Amount ($)']),
        'Message' : messages,
        'Category' : df['Category']
    })

    df_final.to_csv('../data/' + dataset + '_final.csv', index=False)

def main():

    print('Combining training data')
    print('-----------------------')
    combine_outputs('training')
    print()

    print('Combining testing data')
    print('----------------------')
    combine_outputs('testing')
    print()

if __name__ == '__main__':
    title()
    main()
