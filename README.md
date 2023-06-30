# Topic-Modeling-on-News-Articles

# Project Summary -
The BBC News dataset is a collection of news articles from the BBC News website from 2004-2005, consisting of 2225 articles divided into five categories: Business, Entertainment, Politics, Sport, and Tech. The objective of this project is to create an aggregate dataset of all the news articles and perform topic modelling to identify the underlying, latent themes and topics present in the data, among the articles.

Topic modelling is a widely used technique in natural language processing that helps to extract topics from a large collection of documents. The algorithm analyzes the text data and identifies patterns of co-occurring words, thereby grouping similar words into topics. Two popular algorithms used for topic modelling are Latent Dirichlet Allocation (LDA) and Latent Semantic Analysis (LSA). Both algorithms attempt to identify the underlying themes in the dataset by uncovering the latent semantic structure in the text.

In this project, the dataset from individual text files are read and consolidated into a single DataFrame with three columns - Title, Description and the Category. The Description column is carried forward for textual pre-processing, while the Category column provided as input is used to evaluate the accuracy of the model.

Firstly, the LDA algorithm is utilised to perform topic modelling on the BBC News dataset. The dataset is pre-processed by expanding contractions, removing punctuations, digits, whitespaces and stop words. Then Lemmatization is used to group together words to the word's lemma. The pre-processed data is then vectorized using the CountVectorizer from the scikit-learn library. On the vectorized data, the LDA algorithm is applied using Hyperparameter Tuning to identify the underlying topics in the dataset

Additionally, the LDA model is re-applied by using the TfidfVectorizer to vectorize the pre-processed dataset to study the variation in accuracy. A Null Hypotheses that using the CountVectorizer to tokenise input data for an LDA model provides more accurate results, over the TFIDF Vectorizer, is established. An LSA model is also applied on two tokenised inputs of the datasets - one using CountVectorizer and the other using TFIDFVectorizer - as a third model for the project. The LSA algorithm uses the truncated singular value decomposition (SVD) technique to reduce the dimensionality of the dataset and create a topic-term matrix. The topic-term matrix represents the weight of each term in each topic.

After identifying the latent topics, the models are evaluated by verifying whether these topics correspond to the different tags available as input. The identified topics are compared with the original tags for each article by using metrics such as Accuracy and individual topic precisions, recalls and F1 Scores. Finally, word-clouds are generated for each topic to better understand the most frequent words in each topic and whether they actually correspond to that topic or not.

# Problem Statement
The challenge is to perform topic modeling on the BBC News dataset, consisting of news articles from 2004-2005 across five categories. The objective is to uncover the latent themes and topics present within the dataset and evaluate the accuracy of the topic modeling results. This involves selecting an appropriate algorithm, optimizing it, and interpreting the results through visualizations and evaluation metrics. The problem aims to gain insights into the content and categorization of the news articles for further analysis and understanding.

# Variables Description
Each column is a string object, each row describing each article from the BBC News Dataset The columns were

Title - Article title

Description - The article content

Category - Article category/tag (given as input)

The highest number of articles provided were from the Sports category (with 511 articles) and the lowest number belonging to that of Entertainment Category (with 386 articles) (NOTE: these numbers includes duplicate values which will be handled in Data Wrangling section)

![image](https://github.com/pratap-vj/Topic-Modeling-on-News-Articles/assets/123111274/f18f9b36-1efc-4cc7-aa40-68db9f07f253)

![image](https://github.com/pratap-vj/Topic-Modeling-on-News-Articles/assets/123111274/2e170b90-4d7b-4ea6-acc2-d0ae46157fca)

#  Latent Dirichlet Allocation with CountVectorizer

![image](https://github.com/pratap-vj/Topic-Modeling-on-News-Articles/assets/123111274/477249e4-b84f-4c92-9076-f916207c1c3e)

# Latent Dirichlet Allocation with TFIDF Vectorizer

![image](https://github.com/pratap-vj/Topic-Modeling-on-News-Articles/assets/123111274/d238b3f9-8553-4225-9730-cff9beeac28e)

# Latent Semantic Analysis

For Latent Semantic Analysis, the TruncatedSVD algorithm is used. Two models are tried - one with TFIDF Vectorized data as input, and the other with CountVectorized data as input

It can be observed that the LSA algorithm when used on the dataset (tokenised with both TFIDF and the Bag-of-Words), over-determined one particular topic instead of the others.

Model 1: LDA with Count Vectorizer
Log likelihood Score for the LDA model: -441587.0754078079
Perplixity of the LDA model: 743.0567842013797
Accuracy of the LDA model 92.71%

           Topic     Recall  Precision   F1_Score
0       Business  84.294235  97.695853  90.501601
1  Entertainment  92.411924  96.327684  94.329184
2       Politics  98.511166  81.352459  89.113356
3          Sport  98.409543  96.868885  97.633136
4           Tech  90.201729  92.603550  91.386861


Model 2: LDA with TFIDF Vectorizer
Log-likelihood Score for the LDA model: -15266.990851227729
Perplexity of the LDA model: 386.9878030848297
Accuracy of the LDA model 87.25%

           Topic     Recall  Precision   F1_Score
0       Business  88.469185  91.375770  89.898990
1  Entertainment  74.254743  94.809689  83.282675
2       Politics  94.292804  77.079108  84.821429
3          Sport  90.656064  91.017964  90.836653
4           Tech  86.167147  84.225352  85.185185

After analyzing the evaluation metrics for Model 1 (LDA with CountVectorizer) and Model 2 (LSA with TFIDF/CountVectorizer), the following observations can be made:

Perplexity: Model 1 achieves a lower perplexity score compared to Model 2. This suggests that Model 1 performs better in predicting unseen documents, indicating its effectiveness in categorizing textual data.

Log-Likelihood: Model 2 obtains a higher log-likelihood score, indicating a better fit to the observed data. However, it is important to consider additional metrics to assess the overall performance of the models.

Accuracy: Model 1 exhibits a significantly higher accuracy score compared to Model 2, surpassing it by approximately 10%. This implies that Model 1 excels in correctly categorizing articles in proportion to the total number of articles.

Precision and Recall: Model 1 demonstrates better precision and recall scores for each topic compared to Model 2. Although Model 2 achieves high precision in the Entertainment topic, it suffers from low recall, indicating a higher likelihood of missing relevant articles. Model 1 generally outperforms Model 2 in terms of precision, recall, and F1 score across most topics, except for Business and Politics where the scores are comparable.

Based on these results, it is evident that Model 1 (LDA with CountVectorizer) is the preferred choice for topic modeling of this dataset. It exhibits a lower perplexity, high accuracy, and superior precision and recall scores. Therefore, there is insufficient evidence to reject the null hypothesis that CountVectorizer is a better choice than TFIDFVectorizer for tokenizing data input into an LDA model.

In conclusion, Model 1 demonstrates superior performance and is recommended as the optimal choice for accurate and effective topic modeling on this dataset.

# Conclusion
Several key conclusions were drawn from the project:

Encoding errors: During the process of reading News article text files, encoding errors like UnicodeError and ParserError were encountered. To address this issue, exception handling was implemented to ensure proper reading of these articles.

Stopword optimization: The textual pre-processing stage revealed that further optimization of stopwords could be done to improve the model's performance. Certain common words such as "use" and "go" could be removed as they do not contribute significantly to topic identification. However, words like "us" need to be handled carefully as they can have multiple meanings, including "United States."

Stemming vs. lemmatization: Stemming, which reduces words to their root form by chopping off the end, was not implemented in the pre-processing stage. Instead, lemmatization, which considers the context and morphology of words to reduce them to their base form, was used. This decision was made to preserve as much of the original meaning as possible, as topics in topic modeling often rely on language nuances and word usage context.

Choice of tokenization: A null hypothesis was established to determine the optimal choice between CountVectorizer and TFIDFVectorizer for tokenizing the textual data. The findings indicated that CountVectorizer is suitable for tokenization in LDA models. CountVectorizer converts text into an array of word counts, aligning well with the probabilistic nature of LDA, which models word count and topic distributions.

Model comparison: The implemented LDA model using CountVectorizer for tokenization outperformed the LSA model in categorizing the underlying topics in the corpus of articles. The LDA model achieved a high accuracy rate of 93%. Additionally, by examining the word distributions for each topic, a clear correlation between the most frequent words and the topics could be identified, further validating the effectiveness of the LDA model.

In summary, the project highlighted the importance of handling encoding errors, optimizing stopwords, selecting appropriate tokenization methods, and utilizing models like LDA with proper pre-processing techniques to accurately categorize topics within a corpus of articles. The results showcased the superiority of the LDA model with CountVectorizer in achieving high accuracy and identifying meaningful topic distributions.



