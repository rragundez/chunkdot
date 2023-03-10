# pylint: skip-file

from numba.extending import overload, register_jitable
import numpy as np
from numba.np.arraymath import _asarray, valid_kths, nan_aware_less_than
from numba import types


def _partition_factory(pivotimpl, argpartition=False):
    def _partition(A, low, high, I=None):
        mid = (low + high) >> 1
        # NOTE: the pattern of swaps below for the pivot choice and the
        # partitioning gives good results (i.e. regular O(n log n))
        # on sorted, reverse-sorted, and uniform arrays.  Subtle changes
        # risk breaking this property.

        # Use median of three {low, middle, high} as the pivot
        if pivotimpl(A[mid], A[low]):
            A[low], A[mid] = A[mid], A[low]
            if argpartition:
                I[low], I[mid] = I[mid], I[low]
        if pivotimpl(A[high], A[mid]):
            A[high], A[mid] = A[mid], A[high]
            if argpartition:
                I[high], I[mid] = I[mid], I[high]
        if pivotimpl(A[mid], A[low]):
            A[low], A[mid] = A[mid], A[low]
            if argpartition:
                I[low], I[mid] = I[mid], I[low]
        pivot = A[mid]

        A[high], A[mid] = A[mid], A[high]
        if argpartition:
            I[high], I[mid] = I[mid], I[high]
        i = low
        j = high - 1
        while True:
            while i < high and pivotimpl(A[i], pivot):
                i += 1
            while j >= low and pivotimpl(pivot, A[j]):
                j -= 1
            if i >= j:
                break
            A[i], A[j] = A[j], A[i]
            if argpartition:
                I[i], I[j] = I[j], I[i]
            i += 1
            j -= 1
        # Put the pivot back in its final place (all items before `i`
        # are smaller than the pivot, all items at/after `i` are larger)
        A[i], A[high] = A[high], A[i]
        if argpartition:
            I[i], I[high] = I[high], I[i]
        return i

    return _partition


_argpartition_w_nan = register_jitable(_partition_factory(nan_aware_less_than, argpartition=True))


def _select_factory(partitionimpl):
    def _select(arry, k, low, high, idx=None):
        """
        Select the k'th smallest element in array[low:high + 1].
        """
        i = partitionimpl(arry, low, high, idx)
        while i != k:
            if i < k:
                low = i + 1
                i = partitionimpl(arry, low, high, idx)
            else:
                high = i - 1
                i = partitionimpl(arry, low, high, idx)
        return arry[k]

    return _select


_arg_select_w_nan = register_jitable(_select_factory(_argpartition_w_nan))


@register_jitable
def np_argpartition_impl_inner(a, kth_array):
    # allocate and fill empty array rather than copy a and mutate in place
    # as the latter approach fails to preserve strides
    out = np.empty_like(a, dtype=np.int64)

    idx = np.ndindex(a.shape[:-1])  # Numpy default partition axis is -1
    for s in idx:
        arry = a[s].copy()
        idx_arry = np.arange(len(arry))
        low = 0
        high = len(arry) - 1

        for kth in kth_array:
            _arg_select_w_nan(arry, kth, low, high, idx_arry)
            low = kth  # narrow span of subsequent partition

        out[s] = idx_arry
    return out


@overload(np.argpartition)
def np_argpartition(a, kth):
    if not isinstance(a, (types.Array, types.Sequence, types.Tuple)):
        raise TypeError("The first argument must be an array-like")

    if isinstance(a, types.Array) and a.ndim == 0:
        raise TypeError("The first argument must be at least 1-D (found 0-D)")

    kthdt = getattr(kth, "dtype", kth)
    if not isinstance(kthdt, (types.Boolean, types.Integer)):
        # bool gets cast to int subsequently
        raise TypeError("Partition index must be integer")

    def np_argpartition_impl(a, kth):
        a_tmp = _asarray(a)
        if a_tmp.size == 0:
            return a_tmp.copy().astype("int64")
        else:
            kth_array = valid_kths(a_tmp, kth)
            return np_argpartition_impl_inner(a_tmp, kth_array)

    return np_argpartition_impl
