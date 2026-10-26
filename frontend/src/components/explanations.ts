import { LearnMoreProps } from './LearnMore';

/**
 * What every panel on this dashboard is actually measuring.
 *
 * Kept as data in one file rather than inline in each component, for two
 * reasons. The text has to stay consistent about numbers that appear in more
 * than one place -- `tokens / forward pass` is explained here and also
 * referenced by the request-timing panel -- and a single file is where a
 * measurement can be corrected once when it is re-run.
 *
 * Every figure under `measured` comes from a committed benchmark in
 * `benchmarks/results/` or from the test suite, not from an estimate. Where a
 * result is unfavourable it is stated as such: speculative decoding measures
 * *slower* than plain decoding on this hardware, and saying otherwise would
 * make the whole dashboard untrustworthy.
 */

export const TOKENS_PER_PASS: LearnMoreProps = {
  subject: 'tokens / forward pass',
  sections: [
    {
      heading: 'What it is',
      body: [
        'How many output tokens the engine produced per forward pass through the model. It is the single most direct measure of whether continuous batching is working.',
        'A forward pass is the expensive operation — it reads every weight in the model. If four sequences each need a token and the engine runs four passes, it has paid four times for work that one pass could have done.',
      ],
    },
    {
      heading: 'How to read it',
      body: [
        '1.0 means nothing is being batched: one pass, one token. That is the number a naive loop gets, and it is the baseline to beat.',
        'Above 1.0 means several sequences shared a pass. The ceiling is the number of sequences decoding concurrently, so a value near your concurrency level means batching is near-perfect.',
        'It can exceed the concurrency level when speculative decoding is on, because one verification pass can confirm several tokens for the same sequence.',
      ],
    },
    {
      heading: 'How it is measured',
      body: [
        'The engine increments a counter on every call into the model and another on every token emitted; this is their ratio. There is no sampling and no estimation — it counts actual calls.',
        'The denominator includes prefill passes, so the figure dips while a long prompt is being processed and recovers once it is decoding.',
      ],
    },
  ],
  measured: [
    { label: 'Naive sequential loop', value: '1.00' },
    { label: 'Batched, 8 concurrent requests', value: '6.29' },
    { label: 'Live engine end-to-end check', value: '2.77' },
    { label: 'Speculation, batch 4 (CPU)', value: '4.92' },
  ],
};

export const TOKENS_PER_SECOND: LearnMoreProps = {
  subject: 'tokens / second',
  sections: [
    {
      heading: 'What it is',
      body: [
        'Aggregate output rate across every sequence currently decoding, measured over a rolling five-second window rather than since startup.',
      ],
    },
    {
      heading: 'Why the window matters',
      body: [
        'A lifetime average counts idle time, so it understates a live gauge badly: an engine that sat idle for a minute then ran flat out would report a low number while actually being saturated. The five-second window answers "how fast is it going right now", which is the question a dashboard should answer.',
      ],
    },
    {
      heading: 'What it does not tell you',
      body: [
        'Throughput and latency are different goals and can move in opposite directions. Adding concurrency raises this number while making each individual request slower. The per-request panel below is where latency lives.',
        'This figure is hardware-bound, so it is not comparable between machines. The ratios — tokens per pass, speedup against a baseline — are the portable numbers.',
      ],
    },
  ],
};

export const RUNNING_SEQUENCES: LearnMoreProps = {
  subject: 'running sequences',
  sections: [
    {
      heading: 'What it is',
      body: [
        'How many requests the scheduler admitted for this step. It changes every step, which is the defining property of continuous batching: a finished request leaves immediately and a new one joins mid-flight, with no static batch to wait for.',
      ],
    },
    {
      heading: 'Why it can drop unexpectedly',
      body: [
        'The scheduler will preempt a running sequence when the KV pool runs short, evicting its blocks so another can proceed. The preempted request is recomputed from scratch on readmission — nothing is lost, but its latency takes the hit. A sawtooth here under load is preemption, not a bug.',
        'It is also capped by max_num_seqs, so a flat line at that cap means the queue is backed up and the limit is binding.',
      ],
    },
    {
      heading: 'How it is measured',
      body: [
        'Reported directly by the C++ scheduler as the length of its running set for the step that just executed. The waiting count beside it is the admission queue.',
      ],
    },
  ],
};

export const KV_POOL_USED: LearnMoreProps = {
  subject: 'KV pool used',
  sections: [
    {
      heading: 'What it is',
      body: [
        'The fraction of physical KV-cache blocks currently allocated. This is PagedAttention: instead of reserving a contiguous slab per request sized to the worst case, memory is handed out in fixed 16-token blocks and a per-sequence table maps logical positions to physical blocks.',
      ],
    },
    {
      heading: 'Why paging is the point',
      body: [
        'Contiguous allocation must reserve for the longest output a request might produce, and almost every request stops earlier — that unused tail is pure waste. Blocks are allocated only as tokens are actually generated, so the waste is at most one partly-filled block per sequence.',
        'This also makes prefix sharing possible: two requests with the same prompt prefix can point at the same physical blocks, which a contiguous layout cannot express.',
      ],
    },
    {
      heading: 'How to read it',
      body: [
        'High utilisation is good — it means memory is being used rather than reserved. Sustained 100% is where preemption begins, so it is the number to watch when latency gets spiky.',
        'Fragmentation is measured separately and is 0% by construction: every block is the same size, so a freed block always fits the next request.',
      ],
    },
  ],
  measured: [
    { label: 'Block size', value: '16 tokens' },
    { label: 'Fragmentation', value: '0.0%' },
    { label: 'Blocks leaked over 110 requests', value: '0' },
  ],
};

export const REQUEST_TIMINGS: LearnMoreProps = {
  subject: 'per-request latency',
  sections: [
    {
      heading: 'The two numbers that matter',
      body: [
        'Time to first token (TTFT) is how long a user waits before anything appears. It is dominated by prefill — processing the whole prompt — so it grows with prompt length.',
        'Inter-token latency (ITL) is the gap between subsequent tokens, which sets how fast the response appears to stream. A good ITL with a bad TTFT feels like a slow start; the reverse feels like stuttering.',
      ],
    },
    {
      heading: 'How these are measured',
      body: [
        'Timed in this browser, from when the request was sent to when each token arrived over the stream. So they include network and server overhead and are end-to-end user-visible latency, not engine-internal timings.',
        'That makes them larger than the engine\'s own numbers, and deliberately so: the engine\'s view omits exactly the overhead a user experiences.',
      ],
    },
    {
      heading: 'What degrades them',
      body: [
        'Concurrency: more sequences sharing each pass means each one waits longer between its own tokens, even though total throughput rises.',
        'A long prompt arriving mid-flight. Without chunked prefill the engine spends one whole step on that prompt and every in-flight request skips a token — measured at 15.96x a request\'s normal gap. Chunked prefill spreads the prompt over steps, and fusion puts the chunk and the decode in one pass.',
      ],
    },
  ],
  measured: [
    { label: 'Worst stall, no chunking', value: '15.96x' },
    { label: 'With chunked prefill', value: '2.54x' },
    { label: 'With chunking + fusion', value: '1.8x–2.2x' },
  ],
};

export const MEMORY_MAP: LearnMoreProps = {
  subject: 'block occupancy map',
  sections: [
    {
      heading: 'What you are looking at',
      body: [
        'One cell per physical block in the KV pool, coloured by which sequence owns it. This is the physical layout, not the logical one.',
      ],
    },
    {
      heading: 'Why the scatter is the feature',
      body: [
        'A single sequence\'s blocks are deliberately not adjacent. The allocator hands out whichever block is free, and the sequence\'s block table translates its logical positions to those scattered physical ones. Attention then gathers from them.',
        'That scatter is what removes fragmentation: because any free block fits any request, memory never ends up with gaps too small to use. A contiguous allocator has to find a run of adjacent space and fails when memory is checkerboarded.',
      ],
    },
    {
      heading: 'How it is measured',
      body: [
        'Read straight out of the C++ allocator each telemetry tick — an array of block id to owning sequence id. It is the allocator\'s actual state, not a model of it.',
      ],
    },
  ],
};

export const PREFIX_SHARING: LearnMoreProps = {
  subject: 'prefix sharing',
  sections: [
    {
      heading: 'What it does',
      body: [
        'Two requests that begin with the same tokens — a shared system prompt, a few-shot preamble — point at the same physical KV blocks instead of each computing and storing its own copy. The prompt is prefilled once and reused.',
      ],
    },
    {
      heading: 'Why the hash has to be chained',
      body: [
        'A block is only reusable if everything before it is also identical, because a block\'s keys and values depend on all preceding tokens through attention. So each block\'s hash mixes in the previous block\'s hash rather than hashing its own tokens alone.',
        'Hashing blocks independently would match a block whose contents were computed from a different history, and serve KV belonging to someone else\'s conversation — fluent, confident, wrong output, and a cross-request information leak.',
      ],
    },
    {
      heading: 'Copy-on-write',
      body: [
        'A shared block must never be written to. When a sequence needs to write into a block that another sequence is reading, the allocator makes it a private copy first and repoints that sequence\'s table. In practice only full blocks are ever shared and decoding only writes the trailing, partly-filled block, so the copy almost never triggers — it is a guard on an invariant, not a hot path.',
      ],
    },
    {
      heading: 'How it is measured',
      body: [
        'The panel is driven by a reverse index from the allocator — block id to every sequence referencing it — so a block shared three ways is visibly shared three ways. The blocks-saved figure is the sum of reference counts minus the number of allocated blocks: how many extra allocations the pool would have had to serve without sharing.',
      ],
    },
  ],
  measured: [
    { label: '8 requests, 512-token shared prompt', value: '~87% blocks saved' },
    { label: 'Sanitizer run hit rate', value: '56%' },
  ],
};

export const SPECULATIVE_TOKENS: LearnMoreProps = {
  subject: 'speculative decoding',
  sections: [
    {
      heading: 'The idea',
      body: [
        'A small, cheap draft model guesses the next K tokens. The large target model then checks all K in a single pass instead of generating them one at a time. Accepted guesses are free tokens; a rejected guess costs the work already done and nothing more.',
        'The colours show provenance: a drafted token that was accepted, a resampled token where the draft was rejected and the target corrected it, and a bonus token — when every guess was accepted, the verification pass yields one extra token for no draft work at all.',
      ],
    },
    {
      heading: 'Why it cannot change the output',
      body: [
        'Acceptance uses rejection sampling against the target\'s own distribution, so the tokens that come out are distributed exactly as the target alone would have produced them. At temperature 0 this is checkable exactly, and the test suite asserts that greedy speculative output is token-identical to greedy target-only output — at every lookahead value, and with a deliberately useless draft.',
        'So speculation is a pure scheduling optimisation. It can waste work; it cannot corrupt the answer.',
      ],
    },
    {
      heading: 'Honest result on this hardware',
      body: [
        'It is slower here. The draft runs K passes serially per sequence per round, and a 360M target against a 135M draft is only about a 2.7x cost ratio — nowhere near enough to pay for four extra draft passes at a ~60% acceptance rate.',
        'What this project did fix is the structural obstacle: verification is batched, so one target pass checks every sequence\'s guesses at once. Before that, speculating over N sequences cost N target passes per round and was strictly worse than not speculating. Composing the two features is now possible; whether it pays depends on the model pair.',
      ],
    },
  ],
  measured: [
    { label: 'Verify passes, batch 4 (batched vs per-sequence)', value: '5 vs 15' },
    { label: 'Sequences verified per target pass', value: '3.0' },
    { label: 'Wall-clock speedup on CPU', value: '0.62x–0.77x (slower)' },
    { label: 'Output vs non-speculative', value: 'token-identical' },
  ],
};

export const ENGINE_TELEMETRY: LearnMoreProps = {
  subject: 'engine telemetry',
  sections: [
    {
      heading: 'Where these come from',
      body: [
        'The engine exposes a single telemetry snapshot — counters and gauges — polled over a WebSocket, with a REST endpoint as fallback. There is no history on the server: every payload is the state right now, and the charts on this page are accumulated in your browser.',
      ],
    },
    {
      heading: 'Counters worth knowing',
      body: [
        'preemptions — sequences evicted under memory pressure and recomputed. Non-zero under load is normal; growing steadily means the pool is too small for the traffic.',
        'cow_copies — copy-on-write copies taken to protect a shared prefix block. Expected to be near zero.',
        'cached_prefix_tokens — prompt tokens that did not need recomputing. Note this counts both prefix-cache hits and chunked-prefill resumption, so it is a work-avoided figure, not a sharing figure; prefix_cache_hits is the one that means sharing.',
        'blockwise_layer_calls — how many attention layers took the block-wise kernel. If block attention is enabled and this is zero, the feature is not actually running.',
      ],
    },
    {
      heading: 'What is verified, and how',
      body: [
        'The correctness bar throughout this project is byte-identical output, not "close enough". A wrong KV read does not crash — it produces fluent, confident, wrong text — so optimisations are tested by asserting the tokens are exactly unchanged.',
        'Tests are also mutation-tested: deliberate bugs are injected to confirm the tests actually fail. That found several tests which passed against broken code, including a whole acceptance path that no test was exercising.',
      ],
    },
  ],
  measured: [
    { label: 'Python tests', value: '398' },
    { label: 'C++ test cases', value: '18' },
    { label: 'Blocks leaked, 180-cycle sanitizer run', value: '0' },
    { label: 'Memory errors under AddressSanitizer', value: '0' },
  ],
};

export const DEVICE_PANEL: LearnMoreProps = {
  subject: 'execution device & memory',
  sections: [
    {
      heading: 'What this answers',
      body: [
        'Whether the engine is really running on the GPU, and how much memory it is holding there. Until the dashboard had this, the telemetry payload carried no device field at all, so the honest answer was "nobody knows".',
      ],
    },
    {
      heading: 'KV pool vs torch allocated',
      body: [
        'The KV pool is what MemForge reserves up front for the paged cache — it is allocated once at startup and never grows, which is what makes memory behaviour predictable under load.',
        'Torch allocated is everything torch holds: model weights, the pool, and transient activations. The gap between the two is essentially the model.',
        'Peak allocated is the high-water mark since the process started. It is the number that tells you how close you came to running out, which an instantaneous reading cannot.',
      ],
    },
    {
      heading: 'Why CPU shows dashes, not zeros',
      body: [
        'These figures come from torch.cuda.max_memory_allocated, which has no CPU equivalent — there is no allocator to interrogate. So on CPU they are reported as null and rendered as a dash.',
        'Showing 0 MB would mean "we measured, and it was nothing", when the truth is "nothing measured it". Filling this in honestly was blocked until a CUDA-enabled torch was installed; before that the project recorded zero and the zero was misleading.',
      ],
    },
    {
      heading: 'Why the dtype matters here',
      body: [
        'The KV cache is the dominant memory cost at long contexts, and it scales directly with dtype: fp16 halves it against fp32. So the precision shown next to the device is a memory fact as much as a numerical one.',
        'The paged layout is what makes that cost predictable in the first place — memory is handed out in fixed blocks as tokens are produced, rather than reserved per request for the longest output it might generate.',
      ],
    },
  ],
};
