#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
destinations_df = pd.read_csv(r"D:\dsbd_recomendation system\Expanded_Destinations.csv")
reviews_df = pd.read_csv("D:\dsbd_recomendation system\Final_Updated_Expanded_Reviews.csv")
userhistory_df = pd.read_csv("D:\dsbd_recomendation system\Final_Updated_Expanded_UserHistory.csv")
users_df = pd.read_csv("D:\dsbd_recomendation system\Final_Updated_Expanded_Users.csv")
users_df = pd.read_csv("D:\dsbd_recomendation system\Final_Updated_Expanded_Users.csv")


# In[2]:


destinations_df


# In[4]:


reviews_df


# In[5]:


reviews_destinations = pd.merge(reviews_df, destinations_df, on='DestinationID', how='inner')
reviews_destinations_userhistory = pd.merge(reviews_destinations,userhistory_df,on='UserID',how='inner')
df = pd.merge(reviews_destinations_userhistory , users_df, on= 'UserID',how='inner')
df


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:


# content based recommendation


# In[10]:


df['features']=df['Type']+" "+df['State']+" "+df['BestTimeToVisit']+" "+df['Preferences']


# In[12]:


# converting text feature into numbers so that machines can understand that by using tfidf vectorization
vectorizer=TfidfVectorizer(stop_words='english')

deatination_features=vectorizer.fit_transform(df['features'])

deatination_features.toarray()


# In[13]:


# once vector formed now we will find simillarity among all vectors for forming contnet based recommendation system

cosine_sim=cosine_similarity(deatination_features,deatination_features)
cosine_sim


# In[ ]:


# 1.        , 0.30209204, 0.80314548, ..., 0.38161206, 0.10628488,
#         0.10628488], iska mtlb hai haar item ka haar item ke sath similarity ka list hai yeh


# In[21]:



def recommend_destinations(user_id, userhistory_df, destinations_df, cosine_sim):
    """
    Recommend top 5 destinations to a user based on similarity scores.

    Parameters:
    user_id (int): The ID of the user for whom recommendations are generated.
    userhistory_df (DataFrame): User history containing 'UserID' and 'DestinationID'.
    destinations_df (DataFrame): Destination details containing 'DestinationID', 'Name', 'State', 'Type', 'BestTimeToVisit'.
    cosine_sim (ndarray): Precomputed cosine similarity matrix for destinations.

    Returns:
    DataFrame: A DataFrame containing the top 5 recommended destinations.
    """
    
    # Get all destinations that the user has already visited
    visited_destinations = userhistory_df[userhistory_df['UserID'] == user_id]['DestinationID'].values
    
    # If the user has no history, return an empty DataFrame with appropriate columns
    if len(visited_destinations) == 0:
        return pd.DataFrame(columns=['DestinationID', 'Name', 'State', 'Type', 'BestTimeToVisit'])
    
    # Compute similarity scores by summing up similarities of all visited destinations
    # Subtract 1 from visited_destinations to match zero-based indexing of cosine_sim
    similar_scores = np.sum(cosine_sim[visited_destinations - 1], axis=0)
    
    # Sort destinations based on similarity scores in descending order (higher scores first)
    recommended_destinations_idx = np.argsort(similar_scores)[::-1]
    
    recommendations = []  # List to store recommended destinations
    
    # Iterate over sorted destination indices
    for idx in recommended_destinations_idx:
        destination_id = destinations_df.iloc[idx]['DestinationID']
        
        # Only recommend destinations that the user has NOT already visited
        if destination_id not in visited_destinations:
            recommendations.append(
                destinations_df.iloc[idx][['DestinationID', 'Name', 'State', 'Type', 'BestTimeToVisit']].to_dict()
            )
        
        # Stop when we have 5 recommendations
        if len(recommendations) >= 5:
            break

    # Convert the list of recommended destinations to a DataFrame
    return pd.DataFrame(recommendations)

# Example usage:
recommended_destinations = recommend_destinations(1, userhistory_df, destinations_df, cosine_sim)
print(recommended_destinations)


# In[22]:


# Example usage:
recommended_destinations = recommend_destinations(20, userhistory_df, destinations_df, cosine_sim)
print(recommended_destinations)


# In[ ]:




