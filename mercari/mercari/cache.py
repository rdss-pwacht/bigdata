import os

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Protocol

import pandas as pd


class DefaultCacheConfig(NamedTuple):
    enabled: bool
    directory: str


class _CacheConfig(Protocol):
    enabled: bool
    directory: str


def cache_dataframe(
    cache_id: str,
    cache_miss_callback: Callable[..., pd.DataFrame],
    *args: Any,
    **kwargs: Any,
) -> Callable[[_CacheConfig], pd.DataFrame]:
    def factory(cache_config: _CacheConfig) -> pd.DataFrame:
        if cache_config.enabled:
            cache_dir = cache_config.directory
            os.makedirs(cache_dir, exist_ok=True)
            file_path = f"{cache_dir}/{cache_id}.pkl.bz2"
            if os.path.exists(file_path):
                result = pd.read_pickle(file_path)
            else:
                result = cache_miss_callback(*args, **kwargs)
                result.to_pickle(file_path)
        else:
            result = cache_miss_callback(*args, **kwargs)
        return result

    return factory
