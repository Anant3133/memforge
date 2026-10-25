import torch
from memforge_ml.models.paged_attention import PagedKVCache, PagedAttention


def test_paged_kv_write_and_read():
    num_blocks = 8
    block_size = 16
    num_heads = 4
    head_dim = 32

    kv_cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # Simulate sequence with length 25 (requires 2 blocks: 16 in block 0, 9 in block 1)
    seq_len = 25
    block_table = [2, 5]  # Physical blocks assigned

    dummy_keys = torch.randn(1, num_heads, seq_len, head_dim)
    dummy_values = torch.randn(1, num_heads, seq_len, head_dim)

    kv_cache.write_kv(dummy_keys, dummy_values, block_table, start_pos=0)

    # Read back and compare
    read_k, read_v = kv_cache.read_kv(block_table, total_len=seq_len)

    assert torch.allclose(dummy_keys.squeeze(0), read_k, atol=1e-5)
    assert torch.allclose(dummy_values.squeeze(0), read_v, atol=1e-5)


def test_paged_attention_forward():
    num_blocks = 8
    block_size = 16
    num_heads = 2
    head_dim = 16
    context_len = 20
    block_table = [1, 3]

    kv_cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    dummy_keys = torch.randn(1, num_heads, context_len, head_dim)
    dummy_values = torch.randn(1, num_heads, context_len, head_dim)
    kv_cache.write_kv(dummy_keys, dummy_values, block_table, start_pos=0)

    paged_attn = PagedAttention()
    query = torch.randn(1, num_heads, 1, head_dim)  # Single token query

    output = paged_attn(query, kv_cache, block_table, context_len)
    assert output.shape == (1, 1, num_heads * head_dim)
    assert not torch.isnan(output).any()
