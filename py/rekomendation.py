import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ast import literal_eval
import scipy.sparse as sp

class RekomendasiBuku:
  def __init__(self,bookpath,userpath):
    #User
    users_rating_df = pd.read_csv(userpath)
    users_rating_df = users_rating_df[users_rating_df['User_id'].notna()].reset_index(drop=True)
    users_rating_df['Book_Id'] = users_rating_df['Book_Id'].apply(self.id_toInt)
    users_rating_df['User_id'] = users_rating_df['User_id'].apply(self.id_toInt)
    users_rating_df = users_rating_df.drop(['Price', 'profileName', 'review/helpfulness', 'review/time', 'review/summary', 'review/text'],axis=1)

    #Book
    books_data_df = pd.read_csv(bookpath, encoding= 'unicode_escape')
    books_data_df['Book_Id'] = books_data_df['Book_Id'].apply(self.id_toInt)
    self.books_data_df = books_data_df.drop(['publishedDate', 'ratingsCount'],axis=1)

    #Merge Dataset
    users_rating_df = users_rating_df.merge(books_data_df,indicator=True,how='outer')
    users_rating_df = users_rating_df[['User_id','Book_Id','review/score']]
    users_rating_df = users_rating_df.dropna()
    self.users_rating_df = users_rating_df.reset_index()

    #Item-Based
    self.item_matrix = self.users_rating_df.pivot_table(values='review/score', index='Book_Id', columns='User_id')
    self.item_matrix_filled = self.item_matrix.fillna(self.item_matrix.mean(axis=0))
    cos_sim = cosine_similarity(self.item_matrix_filled, self.item_matrix_filled)
    self.item_sim_matrix = pd.DataFrame(cos_sim, index=self.item_matrix.index, columns=self.item_matrix.index)

    #Conten Base Filtering
    tfidf = TfidfVectorizer(stop_words='english')
    self.books_data_df['description'] = self.books_data_df['description'].fillna('')
    self.tfidf_matrix = tfidf.fit_transform(self.books_data_df['description'])
    self.cosine_sim1 = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    self.indices = pd.Series(self.books_data_df.index, index=self.books_data_df['Title'])

    self.books_data_df['category'] = self.books_data_df['categories'].apply(self.get_category)

  def id_toInt(self,x):
    data = ""
    for i in x:
      if i.isnumeric():
        data += i
      else:
        data += str(ord(i))
    return data

  def user_read(self,user_id):
    item = list(self.item_matrix[self.item_matrix[user_id]>0].index)
    df = pd.DataFrame()
    for i in item:
        df = df.append(self.item_sim_matrix[i][:])
    df['total']=df.mean(axis=1)
    df = df.nlargest(15,'total')
    ind = list(df.index)
    read_df = pd.DataFrame() 
    for i in ind:
        read_df = read_df.append(self.books_data_df[self.books_data_df['Book_Id'] == i]) 
    return read_df[['Book_Id','Title', 'categories']]

  def content_recommender(self,df, Title, cosine_sim, indices):
    # Obtain the index of the book that matches the title
    idx = indices[Title]

    # Get the pairwsie similarity scores of all books with that book
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda df: df[1], reverse=True)

    # Get the scores of the 10 most similar books. Ignore the first book.
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return pd.DataFrame(df[['Title','categories','category','image']].iloc[book_indices])

  def get_category(self,x):
    try:
      x=x.lower()
      if len(x)>4:
        return [i.strip() for i in x[2:-2].split(",")]
      else:
        return []
    except:
      return []

  def result(self,titles):
    final_df = pd.DataFrame()
    for i in titles:
        final_df = final_df.append(self.content_recommender(self.books_data_df, i, self.cosine_sim1, self.indices))
    return final_df

  def recommend_books(self,user_id):
    
    sim_books = self.user_read(user_id)
    
    titles = list(sim_books['Title'])
    
    books = self.result(titles)
    
    return books.head()

  def intra_list_similarity(self,predicted, user):

    user['category'] = user['categories'].apply(self.get_category)
    
    feature_df = pd.get_dummies(user['category'].apply(pd.Series).stack()).sum(level=0)

    recs_content = feature_df.loc[predicted]

    similarity = cosine_similarity(X=recs_content.values, dense_output=False)

    upper_right = np.triu_indices(similarity.shape[0], k=1)

    ils_single_user = np.mean(similarity[upper_right])
    
    return ils_single_user