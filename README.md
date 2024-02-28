# ChunkDot

Multi-threaded matrix multiplication and cosine similarity calculations for dense and sparse matrices. Appropriate for calculating the K most similar items for a large number of items by chunking the item matrix representation (embeddings) and using Numba to accelerate the calculations.

Use for:

- [dense embeddings](#dense-embeddings)
- [sparse embeddings](#sparse-embeddings)
- [similarity calculation versus other embeddings](#similarity-calculation-versus-other-embeddings)
- [CosineSimilarityTopK scikit-learn transformer](#cosinesimilaritytopk-scikit-learn-transformer)

## Related blog posts

- [Cosine Similarity for 1 Trillion Pairs of Vectors
](https://pub.towardsai.net/cosine-similarity-for-1-trillion-pairs-of-vectors-11f6a1ed6458)
- [Bulk Similarity Calculations for Sparse Embeddings
](https://pub.towardsai.net/scale-up-bulk-similarity-calculations-for-sparse-embeddings-fb3ecb624727)

## Usage

```bash
pip install -U chunkdot
```

### Dense embeddings

Calculate the 50 most similar and dissimilar items for 100K items.

```python
import numpy as np
from chunkdot import cosine_similarity_top_k

embeddings = np.random.randn(100000, 256)
# using all you system's memory
cosine_similarity_top_k(embeddings, top_k=50)
# most dissimilar items using 20GB
cosine_similarity_top_k(embeddings, top_k=-50, max_memory=20E9)
```
```
<100000x100000 sparse matrix of type '<class 'numpy.float64'>'
 with 5000000 stored elements in Compressed Sparse Row format>
```
```python
# with progress bar
cosine_similarity_top_k(embeddings, top_k=50, show_progress=True)
```
```
100%|███████████████████████████████████████████████████████████████| 129.0/129 [01:04<00:00,  1.80it/s]
<100000x100000 sparse matrix of type '<class 'numpy.float64'>'
  with 5000000 stored elements in Compressed Sparse Row format>
```

Execution time
```python
from timeit import timeit
import numpy as np
from chunkdot import cosine_similarity_top_k

embeddings = np.random.randn(100000, 256)
timeit(lambda: cosine_similarity_top_k(embeddings, top_k=50, max_memory=20E9), number=1)
```
```
58.611996899999994
```

### Sparse embeddings

Calculate the 50 most similar and dissimilar items for 100K items. Items represented by 10K dimensional vectors and an embeddings matrix of 0.005 density.

```python
from scipy import sparse
from chunkdot import cosine_similarity_top_k

embeddings = sparse.rand(100000, 10000, density=0.005)
# using all you system's memory
cosine_similarity_top_k(embeddings, top_k=50)
# most dissimilar items using 20GB
cosine_similarity_top_k(embeddings, top_k=-50, max_memory=20E9)
```
```
<100000x100000 sparse matrix of type '<class 'numpy.float64'>'
 with 5000000 stored elements in Compressed Sparse Row format>
```

Execution time

```python
from timeit import timeit
from scipy import sparse
from chunkdot import cosine_similarity_top_k

embeddings = sparse.rand(100000, 10000, density=0.005)
timeit(lambda: cosine_similarity_top_k(embeddings, top_k=50, max_memory=20E9), number=1)
```
```
51.87472256699999
```
### Similarity calculation versus other embeddings

Given 20K items, for each item, find the 50 most similar items in a collection of other 10K items.

```python
import numpy as np
from chunkdot import cosine_similarity_top_k

embeddings = np.random.randn(20000, 256)
other_embeddings = np.random.randn(10000, 256)

cosine_similarity_top_k(embeddings, embeddings_right=other_embeddings, top_k=10)
```
```
<20000x10000 sparse matrix of type '<class 'numpy.float64'>'
 with 200000 stored elements in Compressed Sparse Row format>
```

 ### CosineSimilarityTopK scikit-learn transformer

Given a pandas DataFrame with 100K rows and

- 2 numerical columns
- 2 categorical columns with 500 categories each

use scikit-learn transformers, the standard scaler for the numerical columns and the one-hot encoder for the categorical columns, to form an embeddings matrix of dimensions 100K x 1002 and then calculate the top 50 most similar rows per each row.

```python
import numpy as np
import pandas as pd

n_rows = 100000
n_categories = 500
df = pd.DataFrame(
    {
        "A_numeric": np.random.rand(n_rows),
        "B_numeric": np.random.rand(n_rows),
        "C_categorical": np.random.randint(n_categories, size=n_rows),
        "D_categorical": np.random.randint(n_categories, size=n_rows),
    }
)
```
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chunkdot import CosineSimilarityTopK

numeric_features = ["A_numeric", "B_numeric"]
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_features = ["C_categorical", "D_categorical"]
categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

cos_sim = CosineSimilarityTopK(top_k=50)

pipe = Pipeline(steps=[("preprocessor", preprocessor), ("cos_sim", cos_sim)])
pipe.fit_transform(df)
```
```
<100000x100000 sparse matrix of type '<class 'numpy.float64'>'
	with 5000000 stored elements in Compressed Sparse Row format>
```

Execution time
```python
from timeit import timeit

timeit(lambda: pipe.fit_transform(df), number=1)
```
```
24.45172154181637
```