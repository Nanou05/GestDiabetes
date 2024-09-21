# -*- coding: utf-8 -*-
"""
Created on Mar  5 17:18:00 2024

@author: N Ouben
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import re 


# import the clean posts

df = pd.read_csv("../data/cleaned_posts.csv", 
                 encoding = 'utf-8', 
                 sep = ';', 
                 decimal = ',')


############################ Feature engineering ######################################

    # adding 2 columns metformin & insulin: 
    # 1 if the post has the word, 0 if it doesn't
df["metformin"] = df["text"].str.contains("metformin") 

df["insulin"] = df["text"].str.contains("insulin")


    # Adding a new column drug: 
        # defining the presence of "metformin" and/or "insulin" in the post
drug = []


for i in range(df.shape[0]) :

	if (df["metformin"][i] & df["insulin"][i]) : 
		drug.append("Metformin and insulin")

	elif df["metformin"][i] :
		drug.append("Only metformin")

	else :
		drug.append("Only insulin")


df["drug"] = drug



############################ VISUALISATION ######################################

# Number of posts per drug

posts_per_drug = df['drug'].value_counts()
drugs = posts_per_drug.index.tolist()
posts = posts_per_drug.tolist()

plt.bar(drugs, posts, color='lightblue')
plt.ylabel('Posts\n')
plt.title('Number of posts per drug')
plt.savefig("../images/posts_per_drug.png")
plt.show()



# Nombre moyen de mots par post par mÃ©dicament

df['word_count'] = df['text'].str.split().apply(len)

drug_word_count = df.groupby('drug')['word_count'].mean()


plt.bar(drug_word_count.index, drug_word_count.tolist(), color='lightblue')
plt.ylabel('Average Word Count per Post')
plt.title('Word count per post per drug')
plt.savefig("../images/mean_word_count_per_drug.png")
plt.show()




# Word clouds

words_to_exclude = ['metformin', 'gestational', 'diabetes', 'insulin', 'week']


	# wordcloud with all the posts  
wordcloud = WordCloud(background_color ='aliceblue',
                      min_font_size = 10, 
                      stopwords = words_to_exclude, 
                      colormap = "Dark2").generate(' '.join(df["text"]))

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("All posts")
plt.savefig("../images/worcloud_all.png")
plt.show()



	# wordcloud "Only metformin" posts 
wordcloud = WordCloud(background_color ='aliceblue',
	min_font_size = 10, 
	stopwords = words_to_exclude, 
	colormap = "Dark2").generate(' '.join(df[df.drug == "Only metformin"]["text"]))
 
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("Only metformin")
plt.savefig("../imagesworcloud_metfo.png")
plt.show()


	# wordcloud "Only insulin" posts 
wordcloud = WordCloud(background_color ='aliceblue',
	min_font_size = 10, 
	stopwords = words_to_exclude, 
	colormap = "Dark2").generate(' '.join(df[df.drug == "Only insulin"]["text"]))
 
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("Only insulin")
plt.savefig("../images/worcloud_insu.png")
plt.show()



	# wordcloud "Metformin and insulin" posts 
wordcloud = WordCloud(background_color ='aliceblue',
	min_font_size = 10, 
	stopwords = words_to_exclude, 
	colormap = "Dark2").generate(' '.join(df[df.drug == "Metformin and insulin"]["text"]))
 
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("Metformin and insulin")
plt.savefig("..images/worcloud_both.png")
plt.show()




############################ ANALYSES ######################################



# Importing the HuggingFace NER model

API_URL = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
headers = {"Authorization": "Bearer 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"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


# Applying the model on the "Metformin and insulin" posts

output = query({
	"inputs": df[df.drug == "Metformin and insulin"]['text'].tolist(),
})


# Extracting the words recognised by the model as signs and symptoms

	## initialization

sign_both = []
related_post_both = []

post_nb = 0

for l in output : 
    # running the model on each post
	nb_sign = 0

	for d in l :
		# running the model on each word


		# Outputting the signs, symptoms and associated posts they were extracted from
       
		try :
			if d['entity_group'] == "Sign_symptom" :
				sign_both.append(d['word'])
				nb_sign = nb_sign + 1

				related_post_both.append(post_nb)


		except :
			nb_sign = nb_sign 

			if nb_sign == 0 :
				sign_both.append("no_sign")
				related_post_both.append(post_nb)		

	post_nb = post_nb + 1

	

	# Storing the output in a dataframe

df_both = pd.DataFrame({'sign' : sign_both, 
                        'post_number' : related_post_both})



# Applying the model on "Only metformin" posts


output = query({
	"inputs": df[df.drug == "Only metformin"]['text'].tolist(),
})


# Extracting the words recognised by the model as signs and symptoms

	## initialization

sign_metfo = []
related_post_metfo = []

post_nb = 0

for l in output :

	nb_sign = 0

	for d in l :

		try :
			if d['entity_group'] == "Sign_symptom" :
				sign_metfo.append(d['word'])
				nb_sign = nb_sign + 1

				related_post_metfo.append(post_nb)

		except :
			nb_sign = nb_sign 

			if nb_sign == 0 :
				sign_metfo.append("no_sign")
				related_post_metfo.append(post_nb)		


	post_nb = post_nb + 1


	## Storing the output in a DataFrame

df_metfo = pd.DataFrame({'sign' : sign_metfo, 
                         'post_number' : related_post_metfo})




# Applying the model on a part of "Only insulin" posts (impossible to run at once on all the data as it is really voluminous)


output = query({
	"inputs": df[df.drug == "Only insulin"]['text'].tolist()[0:df[df.drug == "Only insulin"].shape[0]//2],
})


# Extracting the words recognised by the model as signs and symptoms

	## initialisation

sign_insu = []
related_post_insu = []

post_nb = 0

for l in output :

	nb_sign = 0

	for d in l :


		try :
			if d['entity_group'] == "Sign_symptom" :
				sign_insu.append(d['word'])
				nb_sign = nb_sign + 1

				related_post_insu.append(post_nb)


		except :
			nb_sign = nb_sign 

			if nb_sign == 0 :
				sign_insu.append("no_sign")
				related_post_insu.append(post_nb)		



	post_nb = post_nb + 1


# Applying the model on the remaining part of "Only insulin" posts

output = query({
	"inputs": df[df.drug == "Only insulin"]['text'].tolist()[df[df.drug == "Only insulin"].shape[0]//2:],
})


	## adding the input to the lists generated precedently

for l in output :

	nb_sign = 0

	for d in l :

		try :
			if d['entity_group'] == "Sign_symptom" :
				sign_insu.append(d['word'])
				nb_sign = nb_sign + 1

				related_post_insu.append(post_nb)


		except :
			nb_sign = nb_sign 

			if nb_sign == 0 :
				sign_insu.append("no_sign")
				related_post_insu.append(post_nb)		



	post_nb = post_nb + 1


	## storage in a df

df_insu = pd.DataFrame({'sign' : sign_insu, 
                        'post_number' : related_post_insu})





# Building a dictionnary based on regex for 11 concepts of interest

	## related to: anxiety, fear, stress, etc.

anxiety = [r"[\a-z]*anx[\a-z]*", 
	r"[\a-z]*stress[\a-z]*", 
	r"[\a-z]*nervous[\a-z]*", 
	r"[\a-z]*scar[\a-z]*",
	r"[\a-z]*worr[\a-z]*", 
	r"[\a-z]*panic[\a-z]*", 
	r"[\a-z]*terrif[\a-z]*", 
	r"[\a-z]*freak[\a-z]*", 
	r"[\a-z]*fear[\a-z]*", 
	r"[\a-z]*parano[\a-z]*",
	r"[\a-z]*phobi[\a-z]*"]



	## related to: sadness, depression, hopelessness, guilt, etc.

sadness = [r"[\a-z]*sad[\a-z]*", 
	r"[\a-z]*depres[\a-z]*",
	r"[\a-z]*upset[\a-z]*", 
	r"[\a-z]*cry[\a-z]*",
	r"[\a-z]*cri[\a-z]*", 
	r"[\a-z]*tear[\a-z]*", 
	r"[\a-z]*discourag[\a-z]*", 
	r"[\a-z]*mental breakdown[\a-z]*", 
	r"[\a-z]*lo[\a-z]s[\a-z]* hope[\a-z]*",
	r"[\a-z]*guilt[\a-z]*", 
	r"[\a-z]*defeat[\a-z]*", 
	r"[\a-z]*embarass[\a-z]*",
	r"[\a-z]*asham[\a-z]*"]



	## related to: tiredness, sleep problems, etc.

tired = [r"[\a-z]*tired[\a-z]*", 
	r"[\a-z]*sle[e]*p[\a-z]*", 
	r"[\a-z]*exhaust[\a-z]*", 
	r"[\a-z]*drain[\a-z]*", 
	r"[\a-z]*dizzy[\a-z]*", 
	r"[\a-z]*insomnia[\a-z]*", 
	r"[\a-z]*fatigue[\a-z]*"]


	## related to: hunger, appetite, etc.

eat = [r"[\a-z]*crav[\a-z]*", 
	r"[\a-z]*starv[\a-z]*", 
	r"[\a-z]*hung[\a-z]*", 
	r"[\a-z]*eat[\a-z]*", 
	r"[\a-z]*food[\a-z]*", 
	r"[\a-z]*apetite[\a-z]*"]


	## related to: pain

pain = [r"[\a-z]*pain[\a-z]*", 
	r"[\a-z]*hurt[\a-z]*", 
	r"[\a-z]*harm[\a-z]*", 
	r"[\a-z]*ache[\a-z]*"]


	## related to: digestive system

digestive_issues = [r"[\a-z]*stomach[\a-z]*", 
	r"[\a-z]*diarrhea[\a-z]*", 
	r"[\a-z]*constipat[\a-z]*", 
	r"[\a-z]*pee[\a-z]*", 
	r"[\a-z]*heartburn[\a-z]*", 
	r"[\a-z]*stool[\a-z]*", 
	r"[\a-z]*tummy[\a-z]*"]


	## related to: hypertension, lood pressure

hypertension = [r"[\a-z]*tension[\a-z]*", r"press[\a-z]*"]


	## related to: bleeding

bleeding = [r"[\a-z]*bleed[\a-z]*"]


	## related to: contractions

contraction = [r"[\a-z]*cramp[\a-z]*", r"[\a-z]*contraction[\a-z]*"]


	## related to: nausea, vomitting, etc.

nausea = [r"[\a-z]*nausea[\a-z]*", r"[\a-z]*thr[o,e]w[\a-z]*", r"[\a-z]*vomit[\a-z]*"]


	## related to: cold

cold = [r"[\a-z]*cold[\a-z]*"]


# Function to indicate if a list of words contains a symptom belonging to a pre-defined class: 1 = present, 0 = absent

def classify_sign(sign_list, dic) : 

	matching = [re.match('|'.join(dic), sign) for sign in sign_list]
	find = []

	for m in matching :
		if m != None :
			find.append(m)

	if len(find)  > 0 :
		return 1
	else : 
		return 0



# Identifying the classes present or not in each post: 1 = present, 0 = absent
    
    ## "Metformin and insulin"

df_both = df_both.groupby('post_number').agg(lambda x : list(x))

df_both['anxiety'] = [classify_sign(x, anxiety) for x in df_both['sign'].tolist()]
df_both['sadness'] = [classify_sign(x, sadness) for x in df_both['sign'].tolist()]
df_both['tired'] = [classify_sign(x, tired) for x in df_both['sign'].tolist()]
df_both['appetite'] = [classify_sign(x, eat) for x in df_both['sign'].tolist()]
df_both['pain'] = [classify_sign(x, pain) for x in df_both['sign'].tolist()]
df_both['digestion'] = [classify_sign(x, digestive_issues) for x in df_both['sign'].tolist()]
df_both['hypertension'] = [classify_sign(x, hypertension) for x in df_both['sign'].tolist()]
df_both['bleeding'] = [classify_sign(x, bleeding) for x in df_both['sign'].tolist()]
df_both['contraction'] = [classify_sign(x, contraction) for x in df_both['sign'].tolist()]
df_both['nausea'] = [classify_sign(x, nausea) for x in df_both['sign'].tolist()]
df_both['cold'] = [classify_sign(x, cold) for x in df_both['sign'].tolist()]


	## "Only metformin"

df_metfo = df_metfo.groupby('post_number').agg(lambda x : list(x))

df_metfo['anxiety'] = [classify_sign(x, anxiety) for x in df_metfo['sign'].tolist()]
df_metfo['sadness'] = [classify_sign(x, sadness) for x in df_metfo['sign'].tolist()]
df_metfo['tired'] = [classify_sign(x, tired) for x in df_metfo['sign'].tolist()]
df_metfo['appetite'] = [classify_sign(x, eat) for x in df_metfo['sign'].tolist()]
df_metfo['pain'] = [classify_sign(x, pain) for x in df_metfo['sign'].tolist()]
df_metfo['digestion'] = [classify_sign(x, digestive_issues) for x in df_metfo['sign'].tolist()]
df_metfo['hypertension'] = [classify_sign(x, hypertension) for x in df_metfo['sign'].tolist()]
df_metfo['bleeding'] = [classify_sign(x, bleeding) for x in df_metfo['sign'].tolist()]
df_metfo['contraction'] = [classify_sign(x, contraction) for x in df_metfo['sign'].tolist()]
df_metfo['nausea'] = [classify_sign(x, nausea) for x in df_metfo['sign'].tolist()]
df_metfo['cold'] = [classify_sign(x, cold) for x in df_metfo['sign'].tolist()]


	## "Only insulin"

df_insu = df_insu.groupby('post_number').agg(lambda x : list(x))

df_insu['anxiety'] = [classify_sign(x, anxiety) for x in df_insu['sign'].tolist()]
df_insu['sadness'] = [classify_sign(x, sadness) for x in df_insu['sign'].tolist()]
df_insu['tired'] = [classify_sign(x, tired) for x in df_insu['sign'].tolist()]
df_insu['appetite'] = [classify_sign(x, eat) for x in df_insu['sign'].tolist()]
df_insu['pain'] = [classify_sign(x, pain) for x in df_insu['sign'].tolist()]
df_insu['digestion'] = [classify_sign(x, digestive_issues) for x in df_insu['sign'].tolist()]
df_insu['hypertension'] = [classify_sign(x, hypertension) for x in df_insu['sign'].tolist()]
df_insu['bleeding'] = [classify_sign(x, bleeding) for x in df_insu['sign'].tolist()]
df_insu['contraction'] = [classify_sign(x, contraction) for x in df_insu['sign'].tolist()]
df_insu['nausea'] = [classify_sign(x, nausea) for x in df_insu['sign'].tolist()]
df_insu['cold'] = [classify_sign(x, cold) for x in df_insu['sign'].tolist()]



# Calculating the number of posts mentioning each class of symptoms

print("\n\nMetformin et insulin (" + str(df[df.drug == "Metformin and insulin"].shape[0]) +" posts)")


print ("Anxiety : " + str(sum(df_both.anxiety)))
print ("Sadness : " + str(sum(df_both.sadness)))
print ("Tired : " + str(sum(df_both.tired)))
print ("Appetite : " + str(sum(df_both.appetite)))
print ("Pain : " + str(sum(df_both.pain)))
print ("Digestion : " + str(sum(df_both.digestion)))
print ("Hypertension : " + str(sum(df_both.hypertension)))
print ("Bleedin : " + str(sum(df_both.bleeding)))
print ("Contraction : " + str(sum(df_both.contraction)))
print ("Nausea : " + str(sum(df_both.nausea)))
print ("Cold : " + str(sum(df_both.cold)))



print("\n\nOnly metformin (" + str(df[df.drug == "Only metformin"].shape[0]) +" posts)")


print ("Anxiety : " + str(sum(df_metfo.anxiety)))
print ("Sadness : " + str(sum(df_metfo.sadness)))
print ("Tired : " + str(sum(df_metfo.tired)))
print ("Appetite : " + str(sum(df_metfo.appetite)))
print ("Pain : " + str(sum(df_metfo.pain)))
print ("Digestion : " + str(sum(df_metfo.digestion)))
print ("Hypertension : " + str(sum(df_metfo.hypertension)))
print ("Bleedin : " + str(sum(df_metfo.bleeding)))
print ("Contraction : " + str(sum(df_metfo.contraction)))
print ("Nausea : " + str(sum(df_metfo.nausea)))
print ("Cold : " + str(sum(df_metfo.cold)))


print("\n\nOnly insulin (" + str(df[df.drug == "Only insulin"].shape[0]) +" posts)")



print ("Anxiety : " + str(sum(df_insu.anxiety)))
print ("Sadness : " + str(sum(df_insu.sadness)))
print ("Tired : " + str(sum(df_insu.tired)))
print ("Appetite : " + str(sum(df_insu.appetite)))
print ("Pain : " + str(sum(df_insu.pain)))
print ("Digestion : " + str(sum(df_insu.digestion)))
print ("Hypertension : " + str(sum(df_insu.hypertension)))
print ("Bleedin : " + str(sum(df_insu.bleeding)))
print ("Contraction : " + str(sum(df_insu.contraction)))
print ("Nausea : " + str(sum(df_insu.nausea)))
print ("Cold : " + str(sum(df_insu.cold)))


print("\n\nAll posts (" + str(df.shape[0]) +" posts)")



print ("Anxiety : " + str(sum(df_insu.anxiety) + sum(df_both.anxiety) + sum(df_metfo.anxiety)))
print ("Sadness : " + str(sum(df_insu.sadness) + sum(df_both.sadness) + sum(df_metfo.sadness)))
print ("Tired : "+ str(sum(df_insu.tired) + sum(df_both.tired) + sum(df_metfo.tired)))
print ("Appetite : " + str(sum(df_insu.appetite) + sum(df_both.appetite) + sum(df_metfo.appetite)))
print ("Pain : " + str(sum(df_insu.pain) + sum(df_both.pain) + sum(df_metfo.pain)))
print ("Digestion : " + str(sum(df_insu.digestion) + sum(df_both.digestion) + sum(df_metfo.digestion)))
print ("Hypertension : " + str(sum(df_insu.hypertension) + sum(df_both.hypertension) + sum(df_metfo.hypertension)))
print ("Bleedin : " + str(sum(df_insu.bleeding) + sum(df_both.bleeding) + sum(df_metfo.bleeding)))
print ("Contraction : " + str(sum(df_insu.contraction) + sum(df_both.contraction) + sum(df_metfo.contraction)))
print ("Nausea : " + str(sum(df_insu.nausea) + sum(df_both.nausea) + sum(df_metfo.nausea)))
print ("Cold : " + str(sum(df_insu.cold) + sum(df_both.cold) + sum(df_metfo.cold)))






# Calculating the pecentage of posts mentioning classes of symptoms

df_pourc = pd.DataFrame(
	{'sign_class' : df_both.columns[1:].tolist(),
	'both_posts' : [100 * sum(df_both[:][class_]) / df[df.drug == "Metformin and insulin"].shape[0] for class_ in df_both.columns[1:] ],
	'metfo_posts' : [100 * sum(df_metfo[:][class_]) / df[df.drug == "Only metformin"].shape[0] for class_ in df_metfo.columns[1:] ], 
	'insu_posts' : [100 * sum(df_insu[:][class_]) / df[df.drug == "Only insulin"].shape[0] for class_ in df_insu.columns[1:] ],
	'all_posts' : [100 * (sum(df_both[:][class_]) + sum(df_metfo[:][class_]) + sum(df_insu[:][class_])) / df.shape[0] for class_ in df_insu.columns[1:] ]
	}
).sort_values('all_posts', ascending = False)



# Visualization
plt.bar(df_pourc.sign_class, df_pourc.all_posts, label='all', color ='lightblue')
plt.title('All posts')
plt.ylabel('%\n')
plt.savefig("../images/all_posts_sign_class.png")
plt.show()

fig, axs = plt.subplots(3,1)
axs[0].bar(df_pourc.sign_class, df_pourc.insu_posts,  label='only inuslin', color ='lightblue')
axs[0].title.set_text('Only insulin')
axs[0].set(ylabel = '%\n')
axs[1].bar(df_pourc.sign_class, df_pourc.metfo_posts,  label='only inuslin', color ='lightblue')
axs[1].title.set_text('Only metformin')
axs[1].set(ylabel = '%\n')
axs[2].bar(df_pourc.sign_class, df_pourc.both_posts,  label='only inuslin', color ='lightblue')
axs[2].title.set_text('Metformin and insulin')
axs[2].set(ylabel = '%\n')
fig.tight_layout(pad = 0.5)
plt.savefig("../images/per_drug_posts_sign_class.png")
plt.show()



