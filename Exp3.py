import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
data = {
    'User': ['Alice', 'Alice', 'Bob', 'Carol', 'Carol'],
    'Post': ['Post1', 'Post2', 'Post1', 'Post2', 'Post3'],
    'Rating': [5, 3, 4, 2, 5]
}
df = pd.DataFrame(data)
matrix = df.pivot_table(index='User', columns='Post', values='Rating').fillna(0)
norm = StandardScaler().fit_transform(matrix)
sim = cosine_similarity(norm)
similar_users = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
print("User Similarity Matrix:\n", similar_users)
