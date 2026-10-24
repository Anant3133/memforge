#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memforge/types.hpp"
#include "memforge/sequence.hpp"
#include "memforge/block_allocator.hpp"
#include "memforge/scheduler.hpp"
#include "memforge/prefix_cache.hpp"

namespace py = pybind11;
using namespace memforge;

PYBIND11_MODULE(memforge_c, m) {
    m.doc() = "MemForge C++ Core Engine: High-performance memory allocator and continuous scheduler";

    // SequenceStatus enum
    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("RUNNING", SequenceStatus::RUNNING)
        .value("SWAPPED", SequenceStatus::SWAPPED)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("ABORTED", SequenceStatus::ABORTED)
        .export_values();

    // SequenceStepInfo struct
    py::class_<SequenceStepInfo>(m, "SequenceStepInfo")
        .def(py::init<>())
        .def_readwrite("seq_id", &SequenceStepInfo::seq_id)
        .def_readwrite("block_table", &SequenceStepInfo::block_table)
        .def_readwrite("prompt_length", &SequenceStepInfo::prompt_length)
        .def_readwrite("output_length", &SequenceStepInfo::output_length)
        .def_readwrite("total_length", &SequenceStepInfo::total_length)
        .def_readwrite("status", &SequenceStepInfo::status)
        .def_readwrite("is_prefill", &SequenceStepInfo::is_prefill)
        .def_readwrite("num_cached_tokens", &SequenceStepInfo::num_cached_tokens);

    // SchedulerStepResult struct
    py::class_<SchedulerStepResult>(m, "SchedulerStepResult")
        .def(py::init<>())
        .def_readwrite("running_sequences", &SchedulerStepResult::running_sequences)
        .def_readwrite("finished_sequence_ids", &SchedulerStepResult::finished_sequence_ids)
        .def_readwrite("preempted_sequence_ids", &SchedulerStepResult::preempted_sequence_ids)
        .def_readwrite("aborted_sequence_ids", &SchedulerStepResult::aborted_sequence_ids)
        .def_readwrite("num_waiting", &SchedulerStepResult::num_waiting)
        .def_readwrite("num_free_blocks", &SchedulerStepResult::num_free_blocks)
        .def_readwrite("total_blocks", &SchedulerStepResult::total_blocks)
        .def_readwrite("memory_utilization_ratio", &SchedulerStepResult::memory_utilization_ratio);

    // EngineConfig struct
    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("block_size", &EngineConfig::block_size)
        .def_readwrite("num_gpu_blocks", &EngineConfig::num_gpu_blocks)
        .def_readwrite("max_num_seqs", &EngineConfig::max_num_seqs)
        .def_readwrite("max_model_len", &EngineConfig::max_model_len);

    // Sequence class
    py::class_<Sequence, SequencePtr>(m, "Sequence")
        .def(py::init<SequenceId, std::vector<TokenId>, size_t, size_t>(),
             py::arg("id"), py::arg("prompt_tokens"), py::arg("max_new_tokens"), py::arg("block_size") = DEFAULT_BLOCK_SIZE)
        .def("get_id", &Sequence::get_id)
        .def("get_status", &Sequence::get_status)
        .def("set_status", &Sequence::set_status)
        .def("get_prompt_tokens", &Sequence::get_prompt_tokens)
        .def("get_output_tokens", &Sequence::get_output_tokens)
        .def("get_all_tokens", &Sequence::get_all_tokens)
        .def("get_prompt_len", &Sequence::get_prompt_len)
        .def("get_output_len", &Sequence::get_output_len)
        .def("get_total_len", &Sequence::get_total_len)
        .def("get_block_table", &Sequence::get_block_table)
        .def("append_token", &Sequence::append_token)
        .def("get_num_required_blocks", &Sequence::get_num_required_blocks)
        .def("has_completed_prefill", &Sequence::has_completed_prefill)
        .def("reset_prefill", &Sequence::reset_prefill)
        .def("get_num_cached_tokens", &Sequence::get_num_cached_tokens)
        .def("set_num_cached_tokens", &Sequence::set_num_cached_tokens)
        .def("is_finished", &Sequence::is_finished)
        .def("finish", &Sequence::finish);

    // BlockAllocator class
    py::class_<BlockAllocator, std::shared_ptr<BlockAllocator>>(m, "BlockAllocator")
        .def(py::init<size_t, size_t>(), py::arg("total_blocks"), py::arg("block_size") = DEFAULT_BLOCK_SIZE)
        .def("can_allocate", &BlockAllocator::can_allocate)
        .def("allocate", &BlockAllocator::allocate)
        .def("allocate_blocks", &BlockAllocator::allocate_blocks)
        .def("free", &BlockAllocator::free)
        .def("free_blocks", &BlockAllocator::free_blocks)
        .def("get_num_free_blocks", &BlockAllocator::get_num_free_blocks)
        .def("get_num_allocated_blocks", &BlockAllocator::get_num_allocated_blocks)
        .def("get_total_blocks", &BlockAllocator::get_total_blocks)
        .def("get_block_size", &BlockAllocator::get_block_size)
        .def("free_sequence", &BlockAllocator::free_sequence)
        .def("ensure_capacity", &BlockAllocator::ensure_capacity)
        .def("get_utilization_ratio", &BlockAllocator::get_utilization_ratio)
        .def("reset", &BlockAllocator::reset)
        // Prefix sharing. `release` returns only the ids that actually reached
        // ref_count 0, which the caller must pass to PrefixCache.evict().
        .def("add_ref", &BlockAllocator::add_ref, py::arg("block_id"))
        .def("release", &BlockAllocator::release, py::arg("block_ids"))
        .def("prepare_write", &BlockAllocator::prepare_write,
             py::arg("seq"), py::arg("logical_index"),
             "Copy-on-write guard. Returns (old_block, new_block) when the block "
             "was shared and has been replaced by a private copy the caller must "
             "populate, or None when it was already private.")
        .def("get_ref_count", &BlockAllocator::get_ref_count, py::arg("block_id"))
        .def("get_num_shared_blocks", &BlockAllocator::get_num_shared_blocks)
        .def("get_num_blocks_saved", &BlockAllocator::get_num_blocks_saved);

    // PrefixCache: chained prefix hashing and the hash -> physical registry
    py::class_<PrefixCache, std::shared_ptr<PrefixCache>>(m, "PrefixCache")
        .def(py::init<std::shared_ptr<BlockAllocator>, size_t>(),
             py::arg("allocator"), py::arg("block_size") = DEFAULT_BLOCK_SIZE)
        .def("acquire_shared_prefix", &PrefixCache::acquire_shared_prefix, py::arg("tokens"))
        .def("release_acquired", &PrefixCache::release_acquired, py::arg("block_ids"))
        .def("register_full_blocks", &PrefixCache::register_full_blocks,
             py::arg("tokens"), py::arg("block_table"))
        .def("evict", &PrefixCache::evict, py::arg("released"))
        .def("clear", &PrefixCache::clear)
        .def("num_cached_blocks", &PrefixCache::num_cached_blocks)
        .def("num_hits", &PrefixCache::num_hits)
        .def("num_misses", &PrefixCache::num_misses)
        .def("hit_rate", &PrefixCache::hit_rate)
        .def("get_block_size", &PrefixCache::get_block_size);

    m.def("hash_block", [](uint64_t parent_hash, const std::vector<TokenId>& tokens) {
              return hash_block(parent_hash, tokens.data(), tokens.size());
          },
          py::arg("parent_hash"), py::arg("tokens"),
          "Chained prefix hash of one token block. `parent_hash` is the preceding "
          "block's hash (0 for the first block); chaining is what makes the hash "
          "cover the whole prefix, which is required for correctness. Never 0.");

    // Scheduler class
    py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init<std::shared_ptr<BlockAllocator>, size_t>(), py::arg("allocator"), py::arg("max_num_seqs") = 256)
        .def("add_sequence", &Scheduler::add_sequence)
        .def("step", &Scheduler::step)
        .def("update_sequence", &Scheduler::update_sequence)
        .def("abort_sequence", &Scheduler::abort_sequence)
        .def("get_num_waiting", &Scheduler::get_num_waiting)
        .def("get_num_running", &Scheduler::get_num_running)
        .def("get_num_finished", &Scheduler::get_num_finished)
        .def("has_unfinished_sequences", &Scheduler::has_unfinished_sequences)
        .def("get_sequence", &Scheduler::get_sequence)
        .def("get_allocator", &Scheduler::get_allocator);
}
