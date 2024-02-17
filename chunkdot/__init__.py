import warnings

from chunkdot.chunkdot import chunkdot
from chunkdot.cosine_similarity_top_k import cosine_similarity_top_k
from chunkdot.utils import is_package_installed

if is_package_installed("sklearn"):
    from chunkdot.cosine_similarity_top_k import CosineSimilarityTopK
else:
    warnings.warn(
        "Scikit-Learn not installed. Not importing the ChunkDot Transformer 'CosineSimilarityTopK'"
    )

__version__ = "0.3.0"
__all__ = ["cosine_similarity_top_k", "chunkdot", "CosineSimilarityTopK"]
