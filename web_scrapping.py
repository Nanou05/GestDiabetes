# -*- coding: utf-8 -*-
"""
Created on Mar  5 16:46:09 2024

@author: N Ouben
"""

import requests
import bs4
import pandas as pd



# URLs to scrape
base_url ='https://community.whattoexpect.com/'  

url_topic = 'https://community.whattoexpect.com/forums/gestational-diabetes.html'


n_page = 500 # You can adapt this number to your liking / number of pages available

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



# Function to filter posts containing a specific keyword (insulin or metformin)

def filter_export(posts, keyword, file_path):
    filtered_posts = posts.dropna()
    filtered_posts = filtered_posts[filtered_posts['text'].str.lower().str.contains(keyword)]
    filtered_posts.to_csv(file_path, 
                          encoding="utf-8",
                          index=False,
                          sep=";",
                          decimal=",")
    return filtered_posts

# Load data
posts_diab_gest = pd.read_csv("../data/gd_posts.csv")

# Filter posts containing "metformin" and export
metformin_posts = filter_export(posts_diab_gest, "metformin", "../data/metformin_gd_posts.csv")

# Filter posts containing "insulin" and export
insulin_posts = filter_export(posts_diab_gest, "insulin", "../data/insulin_gd_posts.csv")

# Combine both filtered datasets and remove duplicates
combined_posts = pd.concat([metformin_posts, insulin_posts]).drop_duplicates()

# Export the combined dataset to CSV
combined_posts.to_csv("../data/metfo_insu_gd_posts.csv",
                      encoding="utf-8", 
                      index=False,
                      sep=";",
                      decimal=",")
