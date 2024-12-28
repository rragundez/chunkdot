import warnings

from chunkdot._cosine_similarity_top_k import cosine_similarity_top_k
from chunkdot.chunkdot import chunkdot
from chunkdot.utils import is_package_installed

if is_package_installed("sklearn"):
    # Scikit-learn is a big package and since it is only necessary if using the transformer
    # CosineSimilarityTopK it is not included as an explicit dependency. Therefore, the
    # transformer CosineSimilarityTopK is only loaded if sklearn is already installed by the user.
    from chunkdot._cosine_similarity_top_k import CosineSimilarityTopK
else:
    warnings.warn(
        "Scikit-learn is not installed, therefore chunkdot's sklearn transformer "
        "'CosineSimilarityTopK' will not be loaded. If you plan to use the 'CosineSimilarityTopK' "
        "transformer, please install scikit-learn and re-import chunkdot."
    )

__version__ = "0.6.0"
__all__ = ["cosine_similarity_top_k", "chunkdot", "CosineSimilarityTopK"]
