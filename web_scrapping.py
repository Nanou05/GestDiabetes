# -*- coding: utf-8 -*-
"""
Created on Mar  5 16:46:09 2024

@author: Naima OUBENALI
"""

import requests
import bs4
import pandas as pd



# URLs to scrape
base_url ='https://community.whattoexpect.com/'  

url_topic = 'https://community.whattoexpect.com/forums/gestational-diabetes.html'


n_page = 530 # You can adapt this number to your liking / number of pages available

all_link = []

all_post = []



#Getting the link of each page under the subject "gestational diabetes"

for i in range(n_page) :

	print(i)
	page = requests.get(url_topic, {'page' : i+1 })


	soup =  bs4.BeautifulSoup(page.content, 'html.parser')

	liste_posts = soup.find_all(class_ = "linkDiscussion")


	for post in liste_posts :

		all_link.append(post['href'])




# Exporting the links to a csv file
link_diab_gest = pd.DataFrame({ 
	"link" : all_link
})

link_diab_gest.to_csv("../data/link.csv", 
                      encoding = "utf-8", 
                      index = False, 
                      sep = ";", 
                      decimal = ",")



# Crawling each page that corresponds to a post and getting the textual content of the posts
for link in all_link :

	print(link)

	page = requests.get(base_url + link)

	soup = bs4.BeautifulSoup(page.content, 'html.parser')

	try :
		post = soup.find(class_ = "__messageContent fr-element fr-view").text

	except :
		post = ""
	all_post.append(post)



# Exporting the posts
posts_diab_gest = pd.DataFrame({ 
	"text" : all_post
})


posts_diab_gest.to_csv("../data/gd_posts.csv", 
                       encoding = "utf-8", 
                       index = False, 
                       sep = ";", 
                       decimal = ",")




# Filtering the posts that contain "metformin"

p_filter = posts_diab_gest

	# lowercasing
p_filter.text = p_filter.text.str.lower()
	
	# removal of empty posts
p_filter = p_filter.dropna()
	
	# removal of empty posts
p_filter = p_filter[p_filter['text'].str.contains("metformin")]

	#export the posts to a csv file
p_filter.to_csv("../data/metformin_gd_posts.csv", 
                encoding = "utf-8", 
                index = False, 
                sep = ";", 
                decimal = ",")




# Getting the posts containing the word "insulin"

p_filter2 = posts_diab_gest

	# lowercasing
p_filter2.text = p_filter2.text.str.lower()

	# removal of empty posts
p_filter2 = p_filter2.dropna()

	# filter on "insulin"
p_filter2 = p_filter2[p_filter2['text'].str.contains("insulin")]

	# export to a csv file
p_filter2.to_csv("../data/insulin_gd_posts.csv", 
                 encoding = "utf-8", 
                 index = False, 
                 sep = ";", 
                 decimal = ",")



# Getting the posts that have at least one of the two words "metformin" or "insulin"

p_final = pd.concat([p_filter, p_filter2])
	
	# removal of duplicates
p_final_dedoublonnage = p_final.drop_duplicates(keep = 'first')

	#exporting the data to a csv file
p_final_dedoublonnage.to_csv("../data/metfo_insu_gd_posts.csv", 
                             encoding = "utf-8", 
                             index = False, 
                             sep = ";", 
                             decimal = ",")




