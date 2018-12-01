# Packages
###############################################################################
import numpy as np, pandas as pd, bs4 as bs, matplotlib.pyplot as plt, seaborn as sns
import nltk, urllib.parse, urllib.request, string, gc, random
from nltk.corpus import stopwords
from urllib.error import URLError
from bs4 import BeautifulSoup as bs
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
save_folder = '.../interviews/'
save_prefix = 'interview_remarks'

# Define Functions
###############################################################################
def seconds_to_time(sec):
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))

def semi_rand_intervals_revamp(min_time, max_time, n_nums):
    """random intervals of time between requests"""
    return [i for i in np.random.choice(np.linspace(min_time, max_time, 1000), n_nums)]

def multiple_semi_rand_distr(min_time_list, max_time_list, n_num_list):
    """random intervals of time - multiple distributions"""
    temp_list = []
    for i, x in enumerate(min_time_list):
        temp_list.append(semi_rand_intervals_revamp(min_time_list[i],
                                                    max_time_list[i],
                                                    n_num_list[i]))
    return [i for s in temp_list for i in s]

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

def num_divis_by(num, div_by):
    return ((num // div_by) * div_by) == ((num / div_by) * div_by)

def phantom_scrape_multiple_pause(phan_path, url_list, xpath_list, pause_every, for_sec_min, for_sec_max, fail_sleep = 20):
    """applies 'phantom_scrape' function to list of URLs with semi-random pauses"""
    pause_times = semi_rand_intervals_revamp(for_sec_min, for_sec_max, len(url_list))
    plholder = []
    for i, x in tqdm(enumerate(url_list)):
        try:
            imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
            plholder.append(imp_txt)
            time.sleep(pause_times[i])
            if num_divis_by((i+1), pause_every):
                random.sample([i for i in range(for_sec_min, for_sec_max, 1)], 1)
        except:
            time.sleep(fail_sleep)
            try:
                imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
                plholder.append(imp_txt)
                time.sleep(pause_times[i])
                if num_divis_by((i+1), pause_every):
                    random.sample([i for i in range(for_sec_min, for_sec_max, 1)], 1)
            except:
                time.sleep(fail_sleep * 5)
                imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
                plholder.append(imp_txt)
                time.sleep(pause_times[i])
                if num_divis_by((i+1), pause_every):
                    random.sample([i for i in range(for_sec_min, for_sec_max, 1)], 1)
    return plholder

def scroll_link_scrape(phan_path, web_url, max_scroll = 20):
    """scroll infinite page and scrape links"""
    driver = webdriver.PhantomJS(executable_path = phan_path)
    driver.get(web_url)
    pg_length = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    bottom = False
    scroll_tracker = []
    while(bottom == False):
        last_count = pg_length
        time.sleep(5)
        pg_length = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        if last_count == pg_length:
            bottom = True
        elif np.sum(scroll_tracker) >= max_scroll:
            bottom = True
        else:
            scroll_tracker.append(1)
            print('scrolled ' + str(int(np.sum(scroll_tracker))) + ' / ' + str(int(max_scroll)))
    pg_data = bs(driver.page_source)
    temp_list = []
    for a in pg_data.find_all('a', href = True):
        temp_list.append(a['href'])
    return temp_list

def filter_lst_str(lst, req_txt):
    """filter list of strings by str contains <req_txt>"""
    filtered_lst = []
    for i in set(lst):
        if req_txt in i:
            filtered_lst.append(i)
        else:
            pass
    return filtered_lst

def extr_dt_points(lst):
    """extract talking points where previous line is 'Donald Trump'"""
    lst_noblanks = []
    for l in lst:
        if len(l) > 0:
            lst_noblanks.append(l)
        else:
            pass
    plholder = []
    last_line = lst_noblanks[0]
    for x in lst_noblanks[1:]:
        if last_line == 'Donald Trump':
            plholder.append(x)
        else:
            pass
        last_line = x
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
# Scraping
lnks = scroll_link_scrape(phan_path = ph_path,
                          web_url = 'https://factba.se/transcripts/interviews',
                          max_scroll = 10000)
    
filtered_lnks = filter_lst_str(lst = lnks, req_txt = 'factba.se/transcript/donald-trump-interview')

intv_txt = []
phan_path = ph_path
url_list = filtered_lnks
xpath_list = xpath_list = ['//*[contains(concat( " ", @class, " " ), concat( " ", "speaker-label", " " ))] | //*[(@id = "resultsblock")]//a'] * len(filtered_lnks)
pause_every = 5
for_sec_min = 1
for_sec_max = 3
pause_times = semi_rand_intervals_revamp(for_sec_min, for_sec_max, len(url_list))
plholder = []
for i, x in tqdm(enumerate(url_list[len(intv_txt):])):
    try:
        imp_txt = phantom_scrape(phan_path, url_list[i], xpath_list[i])
        intv_txt.append(imp_txt)
        time.sleep(pause_times[i])
        if num_divis_by((i+1), pause_every):
            random.sample([i for i in range(for_sec_min, for_sec_max, 1)], 1)
    except:
        print('failed on iteration ' + str(i + len(intv_txt)))

# Reformatting
intv_txt_unnested = []
for it in intv_txt:
    if len(it) > 0:
        intv_txt_unnested.append(extr_dt_points(it))
    else:
        pass

# Filter by Length of Remark
intx_txt_unnested_filt = []
for i in intv_txt_unnested:
    for j in i:
        k = rem_transc_notes(j)
        if len(k) >= 30:
            intx_txt_unnested_filt.append(k)

# Saving
###############################################################################
# Save remarks in single doc, line separated
save_line_delim_txt(nested_lines = intx_txt_unnested_filt, save_name = save_folder + save_prefix + '.txt')

# Save individual flat files as a backup
for i, x in tqdm(enumerate(intx_txt_unnested_filt)):
    name = save_folder + 'temp_individual_remarks/remark_' + str(i) + '.txt'
    file = open(name, 'w', encoding="utf-8")
    file.write(x)
    file.close()