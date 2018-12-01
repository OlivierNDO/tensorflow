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
save_folder = '.../vlogs/'
save_prefix = 'vlog_remarks'

# Define Functions
###############################################################################
def scrape_hyperlinks(phan_path, web_url):
    """return href from url"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    temp_list = []
    elems = driver.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        temp_list.append(elem.get_attribute("href"))
    return temp_list

def scrape_hyperlinks_filtered(phan_path, web_url, req_str):
    """return href from url - limited by <string contains>"""
    all_href = scrape_hyperlinks(phan_path = phan_path, web_url = web_url)
    filtered_href = []
    for ah in set(all_href):
        if req_str in ah:
            filtered_href.append(ah)
        else:
            pass
    return filtered_href

def semi_rand_intervals(max_time, n_nums):
    """random intervals of time between requests"""
    return np.random.choice(np.linspace(0, max_time, 1000), n_nums)

def phantom_scrape(phan_path, web_url, x_path):
    """uses phantomJS to scrap url for a given x path"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    time.sleep(semi_rand_intervals(1.5,1))
    tmp_list = []
    for i in driver.find_elements_by_xpath(x_path):
        tmp_list.append(i.text)
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
        inner_txt = rem_transc_notes(' '.join(inner_plholder))
        if len(inner_txt) > 0:
            plholder.append(inner_txt)
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
# Scraping
vlog_urls = scrape_hyperlinks_filtered(phan_path = ph_path,
                                       web_url = 'https://factba.se/trump-vlog',
                                       req_str = 'factba.se/transcript/donald-trump-vlog')

vlog_txt = phantom_scrape_multiple(phan_path = ph_path,
                                   url_list = vlog_urls,
                                   xpath_list = ['//*[(@id = "resultsblock")]//a'] * len(vlog_urls))

# Formatting
vlog_txt_unnested = unnested_lst_lsts(vlog_txt)

# Saving
###############################################################################
# Save remarks in single doc, line separated
save_line_delim_txt(nested_lines = vlog_txt_unnested, save_name = save_folder + save_prefix + '.txt')

# Save individual flat files as a backup
for i, x in tqdm(enumerate(vlog_txt_unnested)):
    name = save_folder + 'temp_individual_remarks/remark_' + str(i) + '.txt'
    file = open(name, 'w', encoding="utf-8")
    file.write(x)
    file.close()