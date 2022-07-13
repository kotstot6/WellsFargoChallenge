
def title():

    print('++++++++++++++++++++')
    print('Script: make_google_searches.py')
    print('Description: web scrapes google\'s knowledge graph to extract a description for each transaction')
    print('Author: Kyle Otstot')
    print('++++++++++++++++++++')
    print()

import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup
import grequests
import requests
from requests_ip_rotator import ApiGateway
import urllib
import csv
import time
import numpy as np

# Parse input

parser = argparse.ArgumentParser(description='Search Google for brand descriptions')

parser.add_argument('--dataset', type=str, default='training_reduced_cleaned.csv', help='CSV file of dataset')
parser.add_argument('--start_index', type=int, default=0, help='index of first row to be processed')
parser.add_argument('--end_index', type=int, default=-1, help='index of last row to be processed')
parser.add_argument('--batch_size', type=int, default=50, help='number of rows processed at a time')

args = parser.parse_args()

gateway = ApiGateway('https://www.google.com', regions=['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2'],
                access_key_id='#########',
                access_key_secret='#######')

gateway.start(force=True)
session = requests.Session()
session.mount("https://www.google.com", gateway)

def main():

    # Read the dataset into memory
    df = pd.read_csv('../data/' + args.dataset)

    # Prepare output CSV file
    output_file = '../data/search_output/' + args.dataset.replace('.', '_') + '-rows-' + str(args.start_index) + '-' + str(args.end_index) + '.csv'

    old_columns = list(df.columns)
    new_columns = (old_columns[:4] + ['Default Brand Description']
            + [old_columns[4]] + ['Coalesced Brand Description'] + old_columns[5:])

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(new_columns)

    train = len(old_columns) > 6

    n_iters = (args.end_index - args.start_index) // args.batch_size

    indices = [(args.start_index + i*args.batch_size, min(args.end_index, args.start_index + (i+1)*args.batch_size))
                        for i in range(n_iters)]

    for i, j in indices:

        amount, m_code, m_cat, def_brand, co_brand, loc = [list(df.values[i:j,k]) for k in range(6)]
        def_brand, co_brand, loc = list(map(str, def_brand)), list(map(str, co_brand)), list(map(str, loc))

        search_dict = search_full(def_brand, co_brand, loc)

        def_desc = [str(result) for result in search_dict['def']]
        co_desc = [str(result) for result in search_dict['co']]

        output_data = [amount, m_code, m_cat, def_brand, def_desc, co_brand, co_desc, loc]

        if train:
            output_data.append(list(df.values[i:j,6]))

        M, N = len(output_data[0]), len(output_data)

        with open(output_file, 'a') as f:
            writer = csv.writer(f)
            for r in range(M):
                writer.writerow([output_data[c][r] for c in range(N)])

        print('Finished rows', i, '-', j , 'of', args.end_index)


# Clean up strings of text

def fix_entry(entry, nums=[str(i) for i in range(10)]):

    if pd.isnull(entry):
        return entry

    entry_list = [e for e in entry.split(' ') if all([n not in e for n in nums]) and e != '']

    if not entry_list:
        return pd.NA

    return (' '.join(entry_list)).lower()

# Format query url

def get_url(q, near=None):

    params = {'q' : q}
    if near is not None:
        params['near'] = near

    q_enc = urllib.parse.urlencode(params)

    return 'https://google.com/search?' + q_enc

# Make request and parse html

def google_search(queries, near=None):

    urls = [get_url(q, near=near) for q in queries]
    rs = (grequests.get(u, session=session) for u in urls)
    responses = grequests.map(rs)

    soups = [BeautifulSoup(r.text, features='html.parser') for r in responses]

    divs_list = [soup.find_all("div", {"class": "BNeawe tAd8D AP7Wnd"}) for soup in soups]
    results = [[fix_entry(div.text.split('\n')[-1]) for div in divs] for divs in divs_list]
    return [[r for r in result if pd.notnull(r)] for result in results]

# Search for both brand names + location

def search_full(def_brand, co_brand, location):

    search_dict = {}

    for search, brand in zip(['def', 'co'], [def_brand, co_brand]):

        results = google_search([b + ' ' + l for b, l in zip(brand, location)])

        search_dict[search] = [list(set(result)) for result in results]

    return search_dict


if __name__ == '__main__':
    title()
    main()
    gateway.shutdown()
