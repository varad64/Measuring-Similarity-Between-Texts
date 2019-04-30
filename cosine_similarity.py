import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
s1 = pd.read_csv('Requirements Dataset.csv')
tt = input("Enter Search Text:\n")
with open('log.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('log.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(tt)
        writer.writerows(lines)
s2 = pd.read_csv('log.csv')
d1 = s1 + s2
cv = CountVectorizer()
sparse_matrix=cv.fit_transform(s1)
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,columns=cv.get_feature_names())
df
print(cosine_similarity(df, df))