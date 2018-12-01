# Packages
###############################################################################
import numpy as np, pandas as pd, bs4 as bs, matplotlib.pyplot as plt, seaborn as sns
import nltk, urllib.parse, urllib.request, string
from nltk.corpus import stopwords
from urllib.error import URLError
from bs4 import BeautifulSoup
from selenium import webdriver
import time, re, string
from time import sleep
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

# Phantom JS Path
###############################################################################
ph_path = '.../phantomjs.exe'

# Save Info
###############################################################################
save_folder = '.../rallies/'
save_prefix = 'rally_remarks'

# URLs with Text
###############################################################################
url_list = ['https://factba.se/transcript/donald-trump-speech-maga-rally-murphysboro-il-october-27-2018',
            'https://factba.se/transcript/donald-trump-speech-future-farmers-america-indianapolis-october-27-2018',
            'https://factba.se/transcript/donald-trump-press-gaggle-air-force-one-arrival-october-27-2018',
            'https://factba.se/transcript/donald-trump-vlog-drug-pricing-october-26-2018',
            'https://factba.se/transcript/donald-trump-remarks-2018-young-black-leaders-october-26-2018',
            'https://factba.se/transcript/donald-trump-remarks-drug-pricing-october-25-2018',
            'https://factba.se/transcript/donald-trump-remarks-anniversary-beirut-barracks-october-25-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-mosinee-wi-october-24-2018',
            'https://factba.se/transcript/donald-trump-remarks-state-leadership-conference-october-23-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-elko-nv-october-20-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-mesa-az-october-19-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-mesa-az-october-19-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-missoula-mt-october-18-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-rochester-mn-october-4-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-southaven-ms-october-2-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-johnson-city-tn-september-29-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-wheeling-wv-september-29-2018',
            'https://factba.se/transcript/donald-trump-speech-political-rally-maga-springfield-mo-september-20-2018',
            'https://factba.se/transcript/donald-trump-speech-political-rally-maga-las-vegas-september-20-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-political-rally-billings-montana-september-6-2018',
            'https://factba.se/transcript/donald-trump-remarks-tea-party-rally-september-9-2015',
            'https://factba.se/transcript/donald-trump-speech-abingdon-va-august-10-2016',
            'https://factba.se/transcript/donald-trump-speech-nashville-march-15-2017',
            'https://factba.se/transcript/donald-trump-speech-rally-harrisburg-pa-april-29-2017',
            'https://factba.se/transcript/donald-trump-speech-rally-cedar-rapids-iowa-june-21-2017',
            'https://factba.se/transcript/donald-trump-speech-celebrate-freedom-rally-july-1-2017',
            'https://factba.se/transcript/donald-trump-speech-rally-youngstown-ohio-july-25-2017',
            'https://factba.se/transcript/donald-trump-speech-rally-huntington-wv-august-3-2017',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-phoenix-arizona-august-22-2017',
            'https://factba.se/transcript/donald-trump-speech-luther-strange-rally-huntsville-alabama-september-22-2017',
            'https://factba.se/transcript/donald-trump-speech-make-america-great-again-pensacola-december-8-2017',
            'https://factba.se/transcript/donald-trump-speech-rally-saccone-pennsylvania-march-10-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-washington-michigan-april-28-2018',
            'https://factba.se/transcript/donald-trump-speech-rally-elkhart-indiana-may-10-2018',
            'https://factba.se/transcript/donald-trump-speech-political-rally-duluth-minnesota-june-20-2018',
            'https://factba.se/transcript/donald-trump-speech-south-carolina-gop-mcmaster-june-23-2018',
            'https://factba.se/transcript/donald-trump-speech-political-rally-north-dakota-june-27-2018',
            'https://factba.se/transcript/donald-trump-speech-make-america-great-again-rally-great-falls-montana-july-5-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-tampa-july-31-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-wilkes-barre-pa-august-2-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-lewis-center-oh-august-4-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-charleston-wv-august-21-2018',
            'https://factba.se/transcript/donald-trump-speech-maga-rally-evansville-indiana-august-30-2018']

xpath_list = ['//*[(@id = "resultsblock")]//a'] * len(url_list)

# Define Functions
###############################################################################
def semi_rand_intervals(max_time, n_nums):
    """random intervals of time between requests"""
    return np.random.choice(np.linspace(0, max_time, 1000), n_nums)

def phantom_scrape(phan_path, web_url, x_path):
    """uses phantomJS to scrap url for a given x path"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    time.sleep(semi_rand_intervals(2,1))
    tmp_list = []
    for i in driver.find_elements_by_xpath(x_path):
        tmp_list.append(i.text)
        time.sleep(semi_rand_intervals(.35,1))
    return tmp_list

def phantom_scrape_multiple(phan_path, url_list, xpath_list):
    """applies 'phantom_scrape' function to list of URLs"""
    plholder = []
    for i, x in tqdm(enumerate(url_list)):
        imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
        plholder.append(imp_txt)
    return plholder

def rem_transc_notes(txt):
    """remove transcriber notes, e.g. (<text>) | [<text>]"""
    return re.sub("[\(\[].*?[\)\]]", "", txt)

def unnested_lst_lsts(lol):
    """returns single string from nested lists of strings"""
    plholder = []
    for i in lol:
        inner_plholder = []
        for j in i:
            if len(j) > 0:
                inner_plholder.append(j)
            else:
                pass
        plholder.append(rem_transc_notes(' '.join(inner_plholder)))
    return plholder

def save_line_delim_txt(nested_lines, save_name):
    """save strings separated by '\n' in single txt file"""
    agg_lines = '\n'.join(nested_lines)
    file = open(save_name, 'w', encoding="utf-8")
    file.write(agg_lines)
    file.close()
    print(str(len(nested_lines)) + ' lines saved to ' + save_name)

# Execute Scraping & Formatting Functions
###############################################################################
grab_txt = phantom_scrape_multiple(ph_path, url_list, xpath_list)
grab_txt_unnested = unnested_lst_lsts(grab_txt)

# Saving
###############################################################################
# Save remarks in single doc, line separated
save_line_delim_txt(nested_lines = grab_txt_unnested, save_name = save_folder + save_prefix + '.txt')

# Save individual flat files as a backup
for i, x in tqdm(enumerate(grab_txt_unnested)):
    name = save_folder + 'temp_individual_remarks/remark_' + str(i) + '.txt'
    file = open(name, 'w', encoding="utf-8")
    file.write(x)
    file.close()