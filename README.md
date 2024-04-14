Multimodal analysis of sentiment based on tweets via stock price prediction 

Project Overview :
Linear Regression Model used 
Actual vs Predicted Prices graphs are used to analyse
error=Actual prices-predicted prices
Impact of volume on close price
Impact of open price on close price
Day with greatest close percentage change
Day with smallest close percentage change
Day with predicted greatest close percentage change
Handling usermentions and hastags
Removing punctuations, special characters,URLS,words and digits containing digits
Tokenization,Removing Gibberish words,Step word Removal
Pos tags,Spacy,Lemmatization,chunking,N-Grams
TF-IDF Vectorization,occurrence-matrix,co-occurrence-matrix
word cloud 1.unigram cloud 2.bigram cloud 3.trigram cloud
Sentiment Analysis using VADER
PCA & K-MEANS clustering 1.PCA-based clustering  2.PCA-based k-means clustering 
Fuzzy c Means clustering 
Agglomerative Hierarchical clustering(Dendrogram)
Stock Prediction using Sentiment Analysis & Generative Adversarial Network.

Project keyword definition: 
Chunking is a bit of a strange word. The root word is “chunk” which means a “piece” or “part of something”. “Chunking” is the process of grouping things together into larger meaningful “chunks” so they’re easier to remember.
NLTK (Natural Language Toolkit) is a Python library used for natural language processing. One of its modules is the WordNet Lemmatizer, which can be used to perform lemmatization on words.
Lemmatization is the process of reducing a word to its base or dictionary form, known as the lemma. For example, the lemma of the word “cats” is “cat”, and the lemma of “running” is “run”.
Lemmatization techniques in natural language processing (NLP) involve methods to identify and transform words into their base or root forms, known as lemmas. These approaches contribute to text normalization, facilitating more accurate language analysis and processing in various NLP applications.
Fuzzy Clustering is a type of clustering algorithm in machine learning that allows a data point to belong to more than one cluster with different degrees of membership. Unlike traditional clustering algorithms, such as k-means or hierarchical clustering, which assign each data point to a single cluster, fuzzy clustering assigns a membership degree between 0 and 1 for each data point for each cluster.
Hierarchical Agglomerative vs Divisive Clustering 
Divisive clustering is more complex as compared to agglomerative clustering, as in the case of divisive clustering we need a flat clustering method as “subroutine” to split each cluster until we have each data having its own singleton cluster.
Divisive clustering is more efficient if we do not generate a complete hierarchy all the way down to individual data leaves. The time complexity of a naive agglomerative clustering is O(n3) because we exhaustively scan the N x N matrix dist_mat for the lowest distance in each of N-1 iterations. Using priority queue data structure we can reduce this complexity to O(n2logn). By using some more optimizations it can be brought down to O(n2). Whereas for divisive clustering given a fixed number of top levels, using an efficient flat algorithm like K-Means, divisive algorithms are linear in the number of patterns and clusters.
A divisive algorithm is also more accurate. Agglomerative clustering makes decisions by considering the local patterns or neighbor points without initially taking into account the global distribution of data. These early decisions cannot be undone. whereas divisive clustering takes into consideration the global distribution of data when making top-level partitioning decisions.

Abstract overview:
Stock market prediction has been an active area of research for a considerable period. Arrival of computing, followed by Machine Learning has upgraded the speed of research as well as opened new avenues. As part of this research study, we aimed to predict the future stock movement of shares using the historical prices aided with availability of sentiment data retrieved from twitter tweets data.  To analyze the sentiments of tweets we have used K-Means clustering algorithm, Linear Regression Analysis and PCA. Also using Fuzzy C-mean Clustering is implemented to create a prediction model.  This sentiment analysis is based on the tweets data retrieval from different sources and then classifying the user perspectives in different classifications accuracy and two unique sentiments (positive and negative).The process of sentiment analysis is the computational procedure that deciding the output by the proposed method is positive or negative. In the proposed technique polarity of each tweet is calculated to differentiate whether the tweet is neutral, positive or negative. The sentiment sentence's polarity is the emotions of user such as angry, sad, happy and joy. A fuzzy-based approach is used to classify user satisfaction according to the relevance of the emotional categories of pleasant and unpleasant. We show that our emotion detection method refines service feature pleasure assessments expressed on scales by users in their reviews. In this work experimental results indicate that fuzzy C-mean clustering  achieves the best accuracy
