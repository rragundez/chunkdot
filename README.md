# ChunkDot

Multi-threaded matrix multiplication and cosine similarity calculations. Appropriate for the calculation of the K most similar items for a large number of items (~1 Million) by partitioning the item matrix representation (embeddings) and using Numba to accelerate the calculations.

## Usage

```bash
pip install -U chunkdot
```

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

## The execution time

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
