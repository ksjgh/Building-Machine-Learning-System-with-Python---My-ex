from sklearn.feature_extraction.text import CountVectorizer


vectorizer=CountVectorizer(min_df=1)
# min_df : if a word frequency <min_df , ignore that word. cf)max_df


content=["How to format my hard disk", "Hard disk format problems"]
X=vectorizer.fit_transform(content)
vectorizer.get_feature_names()
# print(X.toarray().transpose())
