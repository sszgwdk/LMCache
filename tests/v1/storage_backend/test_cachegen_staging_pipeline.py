# SPDX-License-Identifier: Apache-2.0
# Standard
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import (
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import (
    CompressedMemoryAllocator,
    CompressedMemoryObj,
    MemoryFormat,
)
from lmcache.v1.storage_backend.naive_serde.cachegen_decoder import CacheGenDeserializer
from lmcache.v1.storage_backend.naive_serde.cachegen_encoder import CacheGenSerializer

# Local
from tests.v1.utils import dumb_metadata_with_model_name, generate_kv_cache_paged_list_tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vllm_gpu_staging_export_shape_and_metadata():
    num_blocks = 64
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size
    num_tokens = 128

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device="cuda",
        block_size=block_size,
        dtype=torch.bfloat16,
        use_mla=False,
    )
    slot_mapping = torch.tensor(
        random.sample(range(0, num_blocks * block_size), num_tokens),
        device="cuda",
        dtype=torch.int64,
    )

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=False,
        use_mla=False,
    )
    staging_obj = connector.export_staging_tensor(
        0,
        num_tokens,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )
    connector.store_stream.synchronize()

    assert staging_obj.tensor is not None
    assert staging_obj.tensor.is_cuda
    assert staging_obj.metadata.fmt == MemoryFormat.KV_2LTD
    assert staging_obj.tensor.shape == torch.Size([2, num_layers, num_tokens, hidden_dim])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage1_staging_export_rejects_unsupported_paths():
    hidden_dim = 8 * 128
    num_layers = 32

    connector_mla = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=False,
        use_mla=True,
    )
    with pytest.raises(RuntimeError, match="does not support MLA"):
        connector_mla.export_staging_tensor(0, 16)

    connector_layerwise = VLLMPagedMemLayerwiseGPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=False,
        chunk_size=256,
        dtype=torch.bfloat16,
        device="cuda",
        use_mla=False,
    )
    with pytest.raises(RuntimeError, match="does not support layerwise"):
        connector_layerwise.export_staging_tensor(0, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cachegen_gpu_encode_decode_compressed_roundtrip_minimal():
    num_blocks = 64
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size
    num_tokens = 128

    gpu_kv_src = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device="cuda",
        block_size=block_size,
        dtype=torch.bfloat16,
        use_mla=False,
    )
    gpu_kv_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=num_blocks,
        device="cuda",
        block_size=block_size,
        dtype=torch.bfloat16,
        use_mla=False,
    )

    slot_mapping = torch.tensor(
        random.sample(range(0, num_blocks * block_size), num_tokens),
        device="cuda",
        dtype=torch.int64,
    )

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=False,
        use_mla=False,
    )

    staging_obj = connector.export_staging_tensor(
        0,
        num_tokens,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
    )
    connector.store_stream.synchronize()

    config = LMCacheEngineConfig.from_defaults(chunk_size=num_tokens)
    metadata = dumb_metadata_with_model_name(
        model_name="test_model",
        fmt="vllm",
        kv_shape=(num_layers, 2, num_tokens, num_heads, head_size),
    )

    compressed_allocator = CompressedMemoryAllocator(total_size=8 * 1024 * 1024)
    serializer = CacheGenSerializer(
        config,
        metadata,
        compressed_allocator=compressed_allocator,
    )
    deserializer = CacheGenDeserializer(config, metadata)

    compressed_obj = serializer.serialize(staging_obj)
    assert isinstance(compressed_obj, CompressedMemoryObj)
    assert compressed_obj.tensor is not None
    assert compressed_obj.tensor.dtype == torch.uint8
    assert not compressed_obj.tensor.is_cuda
    assert compressed_obj.tensor.is_pinned()

    decoded_obj = deserializer.deserialize(compressed_obj)
    assert decoded_obj.tensor is not None
    decoded_tensor = decoded_obj.tensor
    assert decoded_tensor.is_cuda
    assert decoded_obj.metadata.fmt == MemoryFormat.KV_2LTD
    assert decoded_tensor.shape == staging_obj.tensor.shape

    connector.to_gpu(
        decoded_obj,
        0,
        num_tokens,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    recovered_obj = connector.export_staging_tensor(
        0,
        num_tokens,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
    )
    connector.store_stream.synchronize()

    assert recovered_obj.tensor is not None
    assert torch.allclose(recovered_obj.tensor, decoded_tensor)

    compressed_obj.ref_count_down()
    assert compressed_allocator.memcheck()
    compressed_allocator.close()
