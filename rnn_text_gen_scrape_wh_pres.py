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
from collections import OrderedDict

# Phantom JS Path
###############################################################################
ph_path = '.../phantomjs.exe'

# Save Info
###############################################################################
save_folder = '.../wh_pres_brief/'
save_prefix = 'wh_remarks'

# Define Functions
###############################################################################
def concat_nested_strings(nest_str_lst):
    """returns single string from nested lists of strings"""
    plholder = []
    for i in nest_str_lst:
        for j in i:
            plholder.append(j)
    output = ' '.join(plholder)
    return output

def semi_rand_intervals(max_time, n_nums):
    """random intervals of time between requests"""
    return np.random.choice(np.linspace(0, max_time, 1000), n_nums)

def phantom_scrape(phan_path, web_url, x_path):
    """uses phantomJS to scrap url for a given x path"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    time.sleep(semi_rand_intervals(1,1))
    tmp_list = []
    for i in driver.find_elements_by_xpath(x_path):
        tmp_list.append(i.text)
        time.sleep(semi_rand_intervals(.2,1))
    return tmp_list

def phantom_scrape_multiple(phan_path, url_list, xpath_list):
    """applies 'phantom_scrape' function to list of URLs"""
    plholder = []
    for i, x in tqdm(enumerate(url_list)):
        imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
        plholder.append(imp_txt)
    return plholder

def phantom_href_scrape(phan_path, web_url):
    """uses phantomJS to scrap url for href elements"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    time.sleep(semi_rand_intervals(0.15,1))
    elems = driver.find_elements_by_xpath("//a[@href]")
    tmp_list = []
    for elem in elems:
        tmp_list.append(elem.get_attribute("href"))
    web_url_list = [web_url] * len(tmp_list)
    return web_url_list, tmp_list

def wh_pres_brief_stmt_urls(pg_nums = [pn for pn in range(1,100)]):
    base_str = "https://www.whitehouse.gov/briefings-statements/page/"
    temp_list = []
    for pn in pg_nums:
        temp_list.append(base_str + str(int(pn)) + "/")
    return temp_list

def wh_pres_remarks_agg_scrape(phan_path, pg_nums, xpath):
    """pull many press briefing transcripts at once"""
    url_list = wh_pres_brief_stmt_urls(pg_nums = pg_nums)
    remark_list = []
    counter_list = []
    broken_list = []
    for url in tqdm(url_list):
        url_copies, embed_links = phantom_href_scrape(phan_path = phan_path, web_url = url)
        for i, el in enumerate(embed_links):
            if "https://www.whitehouse.gov/briefings-statements/remarks-president" in el:
                try:
                    remark_list.append(phantom_scrape(phan_path = ph_path,
                                                      web_url = el,
                                                      x_path = xpath))
                    counter_list.append(1)
                    print("No. Briefings Scraped: " + str(int(sum(counter_list))))
                except:
                    broken_list.append(el)
    print("scraping failed for " + str(int(len(broken_list))) + " urls:\n")
    print(broken_list)
    return remark_list

def non_pres_intro(txt, omit_intros = ['PRESIDENT DUQUE', 'PRESIDENT AL-SISI', 'PRESIDENT MOON',
                                       'PRESIDENT DUDA', 'PRESIDENT KENYATTA', 'PRESIDENT PEÑA NIETO',
                                       'VICE PRESIDENT PENCE', 'PRESIDENT SANTOS', 'PRESIDENT JUNCKER',
                                       'PRESIDENT NIINISTÖ', 'THE VICE PRESIDENT', 'PRESIDENT BERSET',
                                       'PRESIDENT MACRON','Q']):
    """determine if string begins with '<not president>: etc..."""
    regex = re.compile('[^a-zA-Z]')
    intro_txt = regex.sub('', txt.split(':')[0])
    num_caps = sum(1 for c in intro_txt if c.isupper())
    num_chars = (len(intro_txt) - intro_txt.count(' '))
    if intro_txt.upper() in omit_intros:
        return 'non_pres'
    elif 'Mr. President' in txt:
        return 'non_pres'
    elif len(intro_txt) < 3:
        return 'pres'
    elif num_caps == num_chars:
        return 'non_pres'
    elif txt[0:2].upper() == 'Q ':
        return 'non_pres'
    else:
        return 'pres'
   
def presbrief_omit_str(nest_str_lst, delete_txt = [], min_char_len = 25):
    """returns single string from nested lists of strings omitting speech by non-pres person"""
    plholder = []
    omitted = []
    for i in nest_str_lst:
        for j in i:
            if non_pres_intro(j) == 'non_pres':
                omitted.append([j.replace('THE PRESIDENT:','')])
            elif len(j) < min_char_len:
                omitted.append([j.replace('THE PRESIDENT:','')])
            else:
                new_s = j.replace('THE PRESIDENT:','')
                if len(delete_txt) > 0:
                    for dt in delete_txt:
                        new_s = new_s.replace(dt, '') 
                plholder.append([new_s.replace('  ', ' ')])
    return plholder, omitted
    
def presbrief_unnest_str(nest_str_lst):
    """returns single string from nested lists of strings omitting speech by non-pres person"""
    plholder = []
    omitted = []
    for i in nest_str_lst:
        for j in i:
            if non_pres_intro(j) == 'non_pres':
                omitted.append(j)
            else:
                plholder.append(j)
    output = ' '.join(plholder)
    return output, omitted
    
def save_line_delim_txt(nested_lines, save_name):
    """save strings separated by '\n' in single txt file"""
    unnested_lines = []
    for nl in nested_lines:
        unnested_lines.append(nl[0])
    agg_lines = '\n'.join(unnested_lines)
    file = open(save_name, 'w', encoding="utf-8")
    file.write(agg_lines)
    file.close()
    print(str(len(nested_lines)) + ' lines saved to ' + save_name)

# Execute Scraping & Formatting Functions
###############################################################################
# Scraping
pres_remarks = wh_pres_remarks_agg_scrape(phan_path = ph_path,
                                          pg_nums = [i for i in range(1,350)],
                                          xpath = '//*[contains(concat( " ", @class, " " ), concat( " ", "editor", " " ))]//p')

# Formatting
pres_remarks_unnested, omitted_remarks = presbrief_omit_str(nest_str_lst = pres_remarks,
                                                            delete_txt = ['Applause', 'applause',
                                                                          'inaudible', 'Inaudible',
                                                                          'PRESIDENT TRUMP:', '(As interpreted.)'],
                                                            min_char_len = 30)

# Saving
###############################################################################
# Save remarks in single doc, line separated
save_line_delim_txt(nested_lines = pres_remarks_unnested, save_name = save_folder + save_prefix + '.txt')

# Save individual flat files as a backup
for i, x in enumerate(pres_remarks_unnested):
    name = save_folder + 'temp_individual_remarks/remark_' + str(i) + '.txt'
    file = open(name, 'w', encoding="utf-8")
    file.write(x[0])
    file.close()