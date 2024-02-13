import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from chunkdot import cosine_similarity_top_k


class CosineSimilarityTopK(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        top_K,
        normalize: bool = True,
        max_memory: int = None,
        force_memory: bool = False,
        show_progress: bool = False,
        min_abs_value=0,
    ):
        self.top_K = top_K
        self.normalize = normalize
        self.max_memory = max_memory
        self.force_memory = force_memory
        self.show_progress = show_progress
        self.min_abs_value = min_abs_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        similarity_matrix = cosine_similarity_top_k(
            embeddings=X,
            top_k=self.top_K,
            normalize=self.normalize,
            max_memory=self.max_memory,
            force_memory=self.force_memory,
            show_progress=self.show_progress,
        )
        if self.min_abs_value > 0:
            mask = np.abs(similarity_matrix.data) < self.min_abs_value
            similarity_matrix.data[mask] = 0
            similarity_matrix.eliminate_zeros()
        return similarity_matrix
