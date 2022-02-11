import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

def readTensor_DBLP(file_name, data_path):
    ft = pd.read_pickle(data_path + file_name)
    n_terms = ft.nunique().term
    n_authors = ft.nunique().author
    n_conf = ft.nunique().conf
      
    le = LabelEncoder()
    le.fit(list(set(ft.author)))
    ft['author_enc'] = le.transform(ft['author'])
    le.fit(list(set(ft.term)))
    ft['term_enc'] = le.transform(ft['term'])
    le.fit(list(set(ft.conf)))
    ft['conf_enc'] = le.transform(ft['conf'])
    ft_array = np.array(ft)
    T = np.zeros((n_terms,n_authors,n_conf), dtype = np.int32)
    y = np.zeros(n_authors, dtype = np.int32)
    z = np.zeros(n_conf, dtype = np.int32)

    for i in range(ft_array.shape[0]):
        T[ft_array[i][6],ft_array[i][5],ft_array[i][7]] = 1
        y[ft_array[i][5]] = ft_array[i][3]
        z[ft_array[i][7]] = ft_array[i][4]
    return T, y


def readTensor_ML(file_name, data_path):
    final = pd.read_pickle(data_path + file_name)
    n = np.shape(final.groupby('userID').count())[0]
    m = np.shape(final.groupby('movieID').count())[0]
    l = np.shape(final.groupby('tagID').count())[0]
    T = np.zeros((n,m,l))   
    y = np.zeros(m)
    for index, row in final.iterrows():
        T[row['user_le'], row['movie_le'], row['tag_le']] = 1
        y[row['movie_le']] = row['genre_le']
    return T, y

    
def readTensor_yelp(file_name, data_path, name):
    yelp = pd.read_pickle(data_path + file_name)
    f = open(data_path + 'yelp_vocabulary_' + name + '.txt','r+', encoding = 'latin1')
    v = dict()
    for l in f.readlines():
        a,b = l.replace('\n','').split(',')
        v[a] = int(b)
    f.close()

    b = len(yelp.business_id.unique())
    u = len(yelp.user_id.unique())

    vect = CountVectorizer(vocabulary = v)
    X_TOR = vect.fit_transform(yelp.text)


    T = np.zeros((b,u,X_TOR.shape[1]))
    y = np.zeros(b)

    for h,r in enumerate(yelp.iterrows()):
        i = r[1].b_label
        j = r[1].u_label
        for k in range(X_TOR.shape[1]):
            T[i,j,k] += X_TOR[h,k]
        if r[1].italian == 1:
            y[i] = 0
        elif r[1].chinese == 1:
            y[i] = 1
        elif r[1].mexican == 1:
            y[i] = 2
    return T,y

def readTensor(dataset, data_path):
    file_dict = {'DBLP':'DBLP_final.pkl', 'MovieLens1':'movielens_final_3g_6_2u.pkl', 'MovieLens2':'movielens_final_3g_10.pkl', 'yelpTOR':'yelp_final_TOR.pkl', 'yelpPGH':'yelp_final_PGH.pkl'}
    if dataset == 'yelpTOR':
        name = 'TOR'
        j = 0
        T,y = readTensor_yelp(file_dict[dataset], data_path, name)
    elif dataset == 'yelpPGH':
        name = 'PGH'
        j = 0
        T,y = readTensor_yelp(file_dict[dataset], data_path, name)
    elif dataset == 'DBLP':
        T,y = readTensor_DBLP(file_dict[dataset], data_path)
        j = 1
    else:
        j = 1
        T,y = readTensor_ML(file_dict[dataset], data_path)
    return T,y,j
    

