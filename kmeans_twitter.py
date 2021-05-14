# Tweets Clustering using k-means
# data hosted on: https://ml-assignmentthree.s3.amazonaws.com/everydayhealth.txt

# STEPS
# pre-process the data
# perform the K-means clustering on the tweets varying k at least 5 times and output results
# K-Means is defined as a class and the algorithm is broken down into different functions

import numpy as np
import pandas as pd

class KMeans:

  def __init__(self, k, iterations, tweets):
    self.k = k
    self.iterations = iterations
    self.tweets = tweets
    self.num_tweets = len(tweets.index)
    self.id_all = tweets["num"].to_numpy()        # each tweet is assigned an 'id' based on its row number
    self.tweets_all = tweets["tweet"].to_numpy()  # each tweet is contained in this array
        

  # calculate the jaccard distance using Distance = 1 - (Intersection/Union)
  # round result to three decimals
  def jaccard_dist(self, a, b):
    intersection = list(set(a) & set(b))
    union = list(set(a) | set(b))
    I_len = len(intersection)
    U_len = len(union)
    return round(1 - float(I_len/U_len),3) 

  # compute the clusters
  def compute_clust(self):
    for clust in range(self.k):
      new_id_centroids = (np.zeros(self.k)).astype(int)
      tweets_centroids = (np.empty(self.k)).astype(object)

      #for x in self.id_centroids:
      for x in range(self.k):
        new_id_centroids[x] = self.id_centroids[x]

      #for x in new_id_centroids:
      for x in range(self.k):
        tweets_centroids[x] = self.tweets_all[new_id_centroids[x]]

      # array that stores closest centroid to each tweet (index of array = tweet id and contents = closest centroid)
      clusters = np.zeros(self.num_tweets).astype(int)

      for i in range(self.num_tweets):
        dist = [self.jaccard_dist(self.tweets_all[i], tweets_centroids[j]) for j in range(self.k)]  # calculate jaccard distance
        result = dist.index(min(dist))   # return the position of the minimum distance for i
        clusters[i] = result             # add result to cluster array
    return clusters

  # update centroids function
  def update_cent(self, clusters):
    new_cent = np.zeros(self.k).astype(int) # array of size k that holds values for new centroids

    # iterate through each cluster 
    for i in range(self.k):
      i_positions = np.zeros(self.num_tweets).astype(int) # i positions is a array of tweet ids currently in cluster i
      count = 0
      for j in range(self.num_tweets):
        if(self.clusters[j] == i):    # if the content of clust[j] equals the current cluster
          i_positions[count] = j      # add the index of current tweet to the i positions array
          count += 1

      # find the average of the i positions array to find the new centroid for cluster i
      sum = 0
      for t in range(len(i_positions)):
        sum += i_positions[t]
      avg = round(sum/len(i_positions),0)

      # add the new centroid as the ith centroid to the array
      new_cent[i] = avg
    return new_cent.astype(int)
  
  # SSE function
  # iterate thorugh each cluster and through each tweet in each cluster
  # find the distance squared of each tweet to its centroid
  # add up each of these to calculate the SSE
  def compute_sse(self):
    sse = 0

    for i in range(self.k): # iterate through clusters
      clust_i_error = 0

      for j in range(self.num_tweets):  # iterate through each tweet
        if(self.clusters[j] == i):      # if the tweet belongs to the cluster
          dist = self.jaccard_dist(self.tweets_all[j], self.tweets_all[self.id_centroids[i]]) # distance
          dist_sqrd = pow(dist, 2)  # distance squared
          clust_i_error = clust_i_error + dist_sqrd # error for individual tweet added to total cluster error
      sse = sse + clust_i_error     # add total cluster error to SSE

    return sse

  # KMeans algorithm
  def kmeans(self):
    # create list of centroid ids and assign to first k ids of the entire list
    self.id_centroids = np.zeros(self.k).astype(int)
    c = 0
    while c < self.k:
      self.id_centroids[c] = self.id_all[c]
      c = c + 1

    # Main Loop
    # assign points to nearest centroid and then recompute centroids i times
    for i in range(self.iterations):
      # compute the clusters
      self.clusters = self.compute_clust()

      # compute the new centroids
      updated_centroids = self.update_cent(self.clusters)
      
      # assign new centroids as the current centroids
      self.id_centroids = updated_centroids

    # Results
    # output K value
    print("K Value:")
    print(self.k)

    # compute the SSE
    self.sse = self.compute_sse()
    print("SSE:")
    print(self.sse)

    # output: K value, SSE, Size of each cluster
    print("cluster sizes:")
    for i in range(self.k):
       clust_i_size = 0
       for j in range(self.num_tweets):
        if(self.clusters[j] == i):
           clust_i_size = clust_i_size + 1
       print(i, clust_i_size)
    return

def preprocessData(link):
  df = pd.read_csv(link, names=["tweet"])

  df.dropna() # drop null values (if any)
  df.drop_duplicates() # drop duplicate rows (if any)
  df["tweet"] = df["tweet"].str[50:]
  df["tweet"] = df["tweet"].str.replace('(\@\w+.*?)',"")
  df["tweet"] = df["tweet"].str.replace('(\#\w)',"")
  df["tweet"] = df["tweet"].str.replace('http\S+|www.\S+', "", case=False)
  df["tweet"] = df["tweet"].str.lower()
  df["tweet"] = df['tweet'].str.replace('[^\w\s]',"") #remove punctuation
  df.insert(0, "num", np.arange(len(df)))
  return df

if __name__ == "__main__":
  link = "https://ml-assignmentthree.s3.amazonaws.com/everydayhealth.txt"
  tweets = preprocessData(link)
  alg = KMeans(10, 5, tweets)   # create kMeans object
  alg.kmeans()                  # preform kmeans clustering on the data