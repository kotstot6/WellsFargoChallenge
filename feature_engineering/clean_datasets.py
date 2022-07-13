
def title():

    print('++++++++++++++++++++')
    print('Script: clean_datasets.py')
    print('Description: cleans up text attributes in raw datasets, and adds log-norm + merchant categories')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import pandas as pd
import numpy as np

# For cleaning strings
def fix_entry(entry, nums=[str(i) for i in range(10)]):

    if pd.isnull(entry):
        return entry

    entry_list = [e for e in entry.split(' ') if all([n not in e for n in nums]) and e != '']

    if not entry_list:
        return pd.NA

    return (' '.join(entry_list)).lower()

# For removing the 1's in strings
def remove_ones(entry):

    if pd.isnull(entry):
        return entry

    entry = ''.join(entry.split('1'))
    entry = ' '.join(entry.split('-'))
    entry = entry.strip(' ')

    if entry == '':
        return pd.NA

    return entry.lower()

# Changes column names and removes most attributes
def reduce_df(df_raw):

    df_reduced = df_raw[['amt', 'merchant_cat_code', 'default_brand',
                         'coalesced_brand', 'default_location', 'Category']].rename(
                            columns = {
                                    'amt' : 'Amount ($)',
                                    'merchant_cat_code' : 'Merchant Code',
                                    'default_brand' : 'Default Brand',
                                    'coalesced_brand' : 'Coalesced Brand',
                                    'default_location' : 'Location'
                                }
                            )

    # int to string
    df_reduced['Merchant Code'] = [pd.NA if pd.isnull(entry)
                                   else str(int(entry)) if  len(str(int(entry))) == 4
                                   else '0' + str(int(entry))
                                   for entry in df_reduced['Merchant Code']]

    # Clean up brand names / location
    for column in ['Default Brand', 'Coalesced Brand', 'Location']:
        df_reduced[column] = [remove_ones(entry) for entry in df_reduced[column].astype('string')]

    return df_reduced

# Receive merchant category from merchant dictionary
def add_merchant_category(df):

    df_merchant = pd.read_csv('../data/merchant_dictionary.csv')
    merchant_dict = {str(code) : fix_entry(desc) for code, desc in zip(df_merchant['Code'], df_merchant['Description'])}

    df['Merchant Category'] = [pd.NA if (pd.isnull(entry) or entry not in merchant_dict)
                                             else merchant_dict[entry]
                                             for entry in df['Merchant Code']]


    df = df[['Amount ($)', 'Merchant Code', 'Merchant Category',
                'Default Brand', 'Coalesced Brand', 'Location', 'Category']]

    return df

# Compute log then standardization
def log_norm_amount(df):

    log_amount = np.log(df['Amount ($)'])
    df['Amount ($)'] = (log_amount - log_amount.mean()) / log_amount.std()

    return df

def main():

    datasets = {
        'training' : pd.read_csv('../data/training_raw.csv'),
        'testing' : pd.read_csv('../data/testing_raw.csv')
    }

    # Reduce and clean each train and test sets
    for dataset, df in datasets.items():

        df = reduce_df(df)
        df = add_merchant_category(df)
        df = log_norm_amount(df)

        # Save them
        df.to_csv('../data/' + dataset + '_reduced_cleaned.csv', index=False)

if __name__ == '__main__':
    title()
    main()
