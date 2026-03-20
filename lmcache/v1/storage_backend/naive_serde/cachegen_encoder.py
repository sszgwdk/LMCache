# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_encoder import encode_function
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CompressedMemoryAllocator,
    CompressedMemoryObj,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.naive_serde.cachegen_basics import CacheGenConfig
from lmcache.v1.storage_backend.naive_serde.serde import Serializer

logger = init_logger(__name__)


class CacheGenSerializer(Serializer):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        compressed_allocator: Optional[CompressedMemoryAllocator] = None,
    ):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

        self.kv_shape = metadata.kv_shape
        if compressed_allocator is not None:
            self.compressed_allocator = compressed_allocator
        else:
            extra_cfg = config.extra_config if config.extra_config is not None else {}
            pool_size_bytes = int(
                extra_cfg.get("cachegen_compressed_pool_size_bytes", 64 * 1024 * 1024)
            )
            self.compressed_allocator = CompressedMemoryAllocator(
                total_size=pool_size_bytes
            )

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer : spec.end_layer] = spec.bins
        return ret.cuda()

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj) -> CompressedMemoryObj:
        """
        Serialize a KV_2LTD MemoryObj to compressed pinned CPU memory object.

        Input:
            memory_obj: the memory object to be serialized.

        Returns:
            CompressedMemoryObj: compressed payload in pinned CPU uint8 buffer.
        """

        # TODO(Jiayi): please avoid this copy by directly performing
        # serialization inside gpu connector.
        assert memory_obj.tensor is not None
        tensor = memory_obj.tensor.cuda()

        # Temporary fix for issue #83: encoder will have the default device 0
        # on all the ray workers. Need to set it to the correct device.
        # Also need to figure out why this happens.
        if torch.cuda.current_device() != tensor.device.index:
            torch.cuda.set_device(tensor.device)
        if tensor.device != self.key_bins.device:
            self.key_bins = self.key_bins.to(tensor.device)
        if tensor.device != self.value_bins.device:
            self.value_bins = self.value_bins.to(tensor.device)

        # tensor is [2, num_layers, num_tokens, hidden_size]
        tensor = tensor.view(*tensor.shape[:-1], self.kv_shape[-2], self.kv_shape[-1])
        tensor = tensor.permute([1, 0, 2, 3, 4])

        # TODO(Jiayi): remove hardcoded "2"
        """ expecting a tensor of shape 
        [num_layers, 2, num_tokens, num_heads, head_size] """
        ntokens = tensor.shape[2]
        output_dict = encode_function(
            tensor,
            self.cachegen_config,
            self.key_bins,
            self.value_bins,
            ntokens,
        )
        payload = output_dict.to_bytes()
        payload_size = len(payload)

        compressed_obj = self.compressed_allocator.allocate(
            torch.Size([payload_size]),
            None,
            fmt=MemoryFormat.BINARY,
        )
        if compressed_obj is None:
            raise MemoryError(
                f"Failed to allocate {payload_size} bytes for cachegen payload"
            )
        assert compressed_obj.tensor is not None
        src = torch.tensor(bytearray(payload), dtype=torch.uint8)
        compressed_obj.tensor.copy_(src, non_blocking=False)
        return compressed_obj
