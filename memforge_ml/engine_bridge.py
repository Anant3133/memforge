"""
Engine Bridge: Connects C++ Core Engine (BlockAllocator & Continuous Scheduler)
with the Python PyTorch Execution Layer. Provides pure-Python fallback if C++ extension is not yet compiled.
"""

from typing import List, Dict, Any, Optional
import time

try:
    import memforge_c
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False


class PyFallbackBlockAllocator:
    """Fallback block allocator in pure Python for environments without C++ build."""
    def __init__(self, total_blocks: int = 512, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))
        self.allocated_blocks = set()

    def can_allocate(self, count: int) -> bool:
        return len(self.free_blocks) >= count

    def allocate_blocks(self, count: int) -> List[int]:
        if len(self.free_blocks) < count:
            raise RuntimeError(f"OOM: requested {count} blocks, available {len(self.free_blocks)}")
        allocated = [self.free_blocks.pop(0) for _ in range(count)]
        self.allocated_blocks.update(allocated)
        return allocated

    def free_blocks(self, blocks: List[int]) -> None:
        for b in blocks:
            if b in self.allocated_blocks:
                self.allocated_blocks.remove(b)
                self.free_blocks.append(b)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_utilization_ratio(self) -> float:
        return len(self.allocated_blocks) / self.total_blocks if self.total_blocks > 0 else 0.0


class MemForgeEngineBridge:
    def __init__(self, total_blocks: int = 512, block_size: int = 16, max_seqs: int = 64):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.max_seqs = max_seqs

        if HAS_CPP_ENGINE:
            print("[EngineBridge] Initializing native C++ engine (memforge_c)...")
            self.allocator = memforge_c.BlockAllocator(total_blocks, block_size)
            self.scheduler = memforge_c.Scheduler(self.allocator, max_seqs)
            self.is_cpp = True
        else:
            print("[EngineBridge] C++ extension not found; running with Python fallback engine.")
            self.allocator = PyFallbackBlockAllocator(total_blocks, block_size)
            self.scheduler = None
            self.is_cpp = False

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "free_blocks": self.allocator.get_num_free_blocks(),
            "allocated_blocks": self.total_blocks - self.allocator.get_num_free_blocks(),
            "utilization_ratio": self.allocator.get_utilization_ratio(),
            "block_size": self.block_size,
            "engine_type": "C++ Native" if self.is_cpp else "Python Fallback",
        }
