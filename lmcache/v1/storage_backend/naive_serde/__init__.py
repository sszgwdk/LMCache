# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Tuple

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.naive_serde.cachegen_decoder import CacheGenDeserializer
from lmcache.v1.storage_backend.naive_serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


def CreateSerde(
    serde_type: str,
    metadata: LMCacheEngineMetadata,
    config: LMCacheEngineConfig,
) -> Tuple[Serializer, Deserializer]:
    if serde_type != "cachegen":
        raise ValueError(
            "Stage-1 local compressed CPU tier only supports `cachegen` serde"
        )

    return CacheGenSerializer(config, metadata), CacheGenDeserializer(config, metadata)


__all__ = [
    "Serializer",
    "Deserializer",
    "CreateSerde",
]
