# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:14:21 2021

@author: Andrea
"""

import requests
from bs4 import BeautifulSoup
import pickle


URL = 'https://www.poesiedautore.it/poeti'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

p_list = []
for a in soup.find_all('a', href=True):
    p_list.append(a['href'])
    
p_list_clean = p_list[7:262]

poetries = dict()
print('------------------------------------------------------------------')
print('Starting poetries scraping from https://www.poesiedautore.it/')
print('------------------------------------------------------------------')
for counter, author in enumerate(p_list_clean):
    print("# {} author: {}".format(counter, author))
    
    url_author = 'https://www.poesiedautore.it/' + author
    page_author = requests.get(url_author)
    soup_author = BeautifulSoup(page_author.content, 'html.parser')
    
    url_poetry = []
    for ul in soup_author.find_all('ul', class_='list-group'):
        for li in ul.find_all('li'):
            a = li.find('a')
            url_poetry.append(a['href'])
    
    poetries[author] = []
    for url_p in url_poetry:
        page_poetry = requests.get(url_p)
        soup_poetry = BeautifulSoup(page_poetry.content, 'html.parser')
        
        try: 
            poetry = soup_poetry.find('pre').get_text()    
            poetries[author].append(poetry)
        except:
            print('Something went wrong with author: {}'.format(author))
            
        
print('------------------------------------------------------------------')

word_count = 0
for p in poetries:
    word_count = word_count + len(p.split())

print('''Process finished. 
Total number of authors: {}. 
Total number of poetries: {}.
Total number of words: {}'''.format(len(p_list_clean), len(poetries), word_count))
print('------------------------------------------------------------------')

with open('poestries_dict.pkl', 'wb') as fp:
    pickle.dump(poetries, fp)

print('Poetries saved to file: poestries_dict.pkl')
