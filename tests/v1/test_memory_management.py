# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    CompressedMemoryAllocator,
    CompressedMemoryObj,
    GPUMemoryAllocator,
    HostMemoryAllocator,
    MemoryFormat,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)


def check_allocator(allocator, max_size):
    # 512 * 512 * 4 = 1MB
    data1 = allocator.allocate([512, 512], torch.float)
    assert data1 is not None
    assert data1.tensor.dtype == torch.float
    assert data1.tensor.shape == (512, 512)

    # 1024 * 1024 * 2 = 2MB
    data2 = allocator.allocate([1024, 1024], dtype=torch.bfloat16)
    assert data2 is not None
    assert data2.tensor.dtype == torch.bfloat16
    assert data2.tensor.shape == (1024, 1024)

    # 2048 * 2048 * 1 = 4MB
    data3 = allocator.allocate([2048, 2048], dtype=torch.int8)
    assert data3 is not None
    assert data3.tensor.dtype == torch.int8
    assert data3.tensor.shape == (2048, 2048)

    allocator.free(data2)
    assert data2.tensor is None
    assert allocator.memcheck()

    allocator.free(data1)
    assert data1.tensor is None
    assert allocator.memcheck()

    allocator.free(data2)  # This should not crash

    data4 = allocator.allocate([3, 5, 7], dtype=torch.half)
    assert data4 is not None
    assert data4.tensor.dtype == torch.half
    assert data4.tensor.shape == (3, 5, 7)

    data_fail = allocator.allocate([max_size], dtype=torch.float)  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.free(data1)
    allocator.free(data2)
    allocator.free(data3)
    allocator.free(data4)

    assert allocator.memcheck()

    allocator.close()


def check_paged_allocator(allocator, shape, dtype, fmt, max_num_pages):
    # Allocate one page
    data1 = allocator.allocate(shape, dtype, fmt)
    assert data1 is not None
    assert data1.tensor.dtype == dtype
    assert data1.tensor.shape == shape

    # Allocate another 2 pages
    data2 = allocator.batched_allocate(shape, dtype, 2, fmt)

    for data in data2:
        assert data is not None
        assert data.tensor.dtype == dtype
        assert data.tensor.shape == shape

    # Allocate a smaller page
    smaller_shape = torch.Size([2, 32, 8, 1024])
    data3 = allocator.allocate(smaller_shape, dtype, fmt)
    assert data3 is not None
    assert data3.tensor.dtype == dtype
    assert data3.tensor.shape == smaller_shape

    allocator.free(data3)
    assert allocator.memcheck()

    allocator.batched_free(data2)
    assert allocator.memcheck()

    allocator.free(data1)
    assert allocator.memcheck()

    data_fail = allocator.batched_allocate(
        shape, dtype, max_num_pages + 1, fmt
    )  # This should fail
    assert data_fail is None

    assert allocator.memcheck()

    allocator.close()


@pytest.mark.parametrize(
    "use_paging",
    [True, False],
)
def test_tensor_allocator(use_paging):
    total_size = 1024 * 1024 * 128  # 128MB
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")
    if use_paging:
        shape = torch.Size([2, 32, 16, 1024])  # 64 pages
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD
        num_pages = 64
        allocator = PagedTensorMemoryAllocator(tensor_buffer, shape, dtype, fmt)
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        allocator = TensorMemoryAllocator(tensor_buffer)
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
@pytest.mark.parametrize(
    "use_paging",
    [
        False,
        True,
    ],
)
def test_device_allocators(alloc_cls, use_paging):
    total_size = 1024 * 1024 * 128  # 128MB

    shape = torch.Size([2, 32, 16, 1024])  # 64 pages
    dtype = torch.bfloat16
    fmt = MemoryFormat.KV_2LTD

    allocator = alloc_cls(
        total_size, use_paging=use_paging, shape=shape, dtype=dtype, fmt=fmt
    )

    if use_paging:
        num_pages = 64
        check_paged_allocator(allocator, shape, dtype, fmt, num_pages)
    else:
        check_allocator(allocator, total_size)

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_inplace_modification(alloc_cls):
    total_size = 1024 * 1024
    allocator = alloc_cls(total_size)

    data = allocator.allocate([4096], torch.float)
    assert data is not None
    assert data.tensor.dtype == torch.float
    assert data.tensor.shape == (4096,)

    data.tensor.fill_(1.0)
    assert torch.all(data.tensor == 1.0)

    data.tensor[1] = 2.0
    assert data.tensor[1] == 2.0

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_boundary_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 10], torch.float)
    allocator.allocate([512, 10], torch.float)
    allocator.free(data1)

    # `FreeBlock` with size 0 shouldn't exist in the allocator
    allocator.allocate([512, 10], torch.float)

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        HostMemoryAllocator,
        PinMemoryAllocator,
        GPUMemoryAllocator,
        MixedMemoryAllocator,
    ],
)
def test_batched_alloc(alloc_cls):
    total_size = 32 * 100 * 2 * 1024 * 2
    batch_size = 32
    allocator = alloc_cls(total_size)
    objs = allocator.batched_allocate(
        [100, 2, 1024], torch.bfloat16, batch_size, MemoryFormat.KV_T2D
    )

    assert len(objs) == batch_size
    for obj in objs:
        assert obj is not None
        assert obj.tensor is not None
        assert obj.tensor.dtype == torch.bfloat16
        assert obj.tensor.shape == (100, 2, 1024)
    allocator.batched_free(objs)

    if isinstance(allocator, MixedMemoryAllocator):
        assert len(allocator.pin_allocator.explicit_list) == 1
    else:
        assert len(allocator.allocator.explicit_list) == 1

    allocator.close()


@pytest.mark.parametrize(
    "alloc_cls",
    [
        MixedMemoryAllocator,
    ],
)
def test_mixed_alloc(alloc_cls):
    total_size = 1 << 25
    allocator = alloc_cls(total_size)
    data1 = allocator.allocate([512, 0], None, MemoryFormat.BINARY_BUFFER)
    allocator.allocate([512, 10], torch.float)
    allocator.free(data1)

    assert len(allocator.pin_allocator.explicit_list) == 1

    assert isinstance(data1, BytesBufferMemoryObj)

    assert len(data1.byte_array) == 512

    allocator.close()


def test_compressed_memory_obj_lifecycle():
    allocator = CompressedMemoryAllocator(total_size=512, align_bytes=64)
    try:
        mem_obj = allocator.allocate([128], dtype=None, fmt=MemoryFormat.BINARY)
        assert isinstance(mem_obj, CompressedMemoryObj)
        assert mem_obj is not None
        assert mem_obj.is_valid()
        assert mem_obj.get_size() == 128
        assert mem_obj.get_physical_size() >= 128
        assert mem_obj.can_evict

        mem_obj.pin()
        assert mem_obj.is_pinned
        assert not mem_obj.can_evict

        mem_obj.unpin()
        assert not mem_obj.is_pinned
        assert mem_obj.can_evict

        mem_obj.ref_count_up()
        assert mem_obj.get_ref_count() == 2
        mem_obj.ref_count_down()
        assert mem_obj.get_ref_count() == 1

        mem_obj.ref_count_down()
        assert not mem_obj.is_valid()
        assert allocator.memcheck()
    finally:
        allocator.close()


def test_compressed_allocator_variable_length_alloc_free():
    allocator = CompressedMemoryAllocator(total_size=1024, align_bytes=64)
    try:
        obj1 = allocator.allocate([100], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj1 is not None
        obj1_address = obj1.metadata.address

        obj2 = allocator.allocate([180], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj2 is not None
        assert obj2.metadata.address > obj1_address

        obj1.ref_count_down()

        # First-fit variable-length allocator should reuse the freed head block.
        obj3 = allocator.allocate([96], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj3 is not None
        assert obj3.metadata.address == obj1_address

        obj2.ref_count_down()
        obj3.ref_count_down()
        assert allocator.memcheck()
    finally:
        allocator.close()


def test_compressed_allocator_out_of_capacity():
    allocator = CompressedMemoryAllocator(total_size=256, align_bytes=64)
    try:
        obj1 = allocator.allocate([200], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj1 is not None

        obj2 = allocator.allocate([200], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj2 is None

        # Request larger than max bucket must fail as well.
        obj3 = allocator.allocate([300], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj3 is None

        obj1.ref_count_down()
        assert allocator.memcheck()
    finally:
        allocator.close()


def test_compressed_allocator_coalescing_reuse():
    allocator = CompressedMemoryAllocator(total_size=512, align_bytes=64)
    try:
        obj1 = allocator.allocate([100], dtype=None, fmt=MemoryFormat.BINARY)
        obj2 = allocator.allocate([100], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj1 is not None
        assert obj2 is not None

        obj1.ref_count_down()
        obj2.ref_count_down()

        # Two adjacent freed blocks should be coalesced and satisfy a larger request.
        obj3 = allocator.allocate([192], dtype=None, fmt=MemoryFormat.BINARY)
        assert obj3 is not None
        obj3.ref_count_down()
    finally:
        allocator.close()
