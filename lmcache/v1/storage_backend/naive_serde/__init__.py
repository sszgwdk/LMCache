# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.memory_management import CompressedMemoryAllocator
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.naive_serde.cachegen_decoder import CacheGenDeserializer
from lmcache.v1.storage_backend.naive_serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


def CreateSerde(
    serde_type: str,
    metadata: LMCacheEngineMetadata,
    config: LMCacheEngineConfig,
    compressed_allocator: Optional[CompressedMemoryAllocator] = None,
) -> Tuple[Serializer, Deserializer]:
    if serde_type != "cachegen":
        raise ValueError(
            "Stage-1 local compressed CPU tier only supports `cachegen` serde"
        )

    return (
        CacheGenSerializer(
            config,
            metadata,
            compressed_allocator=compressed_allocator,
        ),
        CacheGenDeserializer(config, metadata),
    )


__all__ = [
    "Serializer",
    "Deserializer",
    "CreateSerde",
]
