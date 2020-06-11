
from flask import Flask, request, jsonify, render_template
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import difflib
import pickle

app = Flask(__name__) #Initialize the flask App
#model = pickle.load(open('r_model.pkl', 'rb'))



#pd.set_option('display.max_columns', 100)
df = pd.read_csv('songs_i_rarely_ever_skip.csv')

df = df[['Artist Name','Track Name','Album Name']]
df= df.rename(columns={'Artist Name':'artist','Track Name':'track','Album Name':'album'})
all_titles = [df['track'][i] for i in range(len(df['track']))]

# discarding the commas between the artists' full names and getting only the first three names
df['artist'] = df['artist'].map(lambda x: x.split(',')[:3])

# putting the album in a list of words
df['album'] = df['album'].map(lambda x: x.lower().split(',')[:4])

# merging together first and last name for each singer, so it's considered as one word 
# and there is no mix up between people sharing a first name
for index, row in df.iterrows():
    row['artist'] = [x.lower().replace(' ','') for x in row['artist']]

df.set_index('track', inplace = True)

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
            words = words + ' '.join(row[col])+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the song titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices[:3]

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim

# function that takes in song title as input and returns the top 10 recommended songs
def recommendations(title, cosine_sim = cosine_sim):
    
    recommended_songs = []
    
    # gettin the index of the song that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar songs
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the titles of the best 10 matching songs
    for i in top_10_indexes:
        recommended_songs.append(list(df.index)[i])
        
    return recommended_songs

recommendations('Toxic')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/main',methods=['POST'])

def main():
    s_name = request.form['song_name']
    if s_name not in all_titles:
        return render_template('negative.html',name=s_name)
    else:
        result_final = recommendations(s_name,cosine_sim = cosine_sim)
        names = []
        for i in range(len(result_final)):
            names=result_final
            return render_template('positive.html',song_names=names,search_name=s_name)

if __name__ == '__main__':
    app.run(debug=True)