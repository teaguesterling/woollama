# Queueing behind a model swap (#39)

## The gap

`Gate` queues requests **for a model** correctly. It does not queue **across a swap**. When
another consumer holds model A on a capacity-1 device and a request arrives for model B, the
second caller gets `503` immediately rather than waiting its turn.

The immediate `503` comes from exactly one place — `pool.rs`, in `ensure_loaded`:

```rust
let victim = resolver::pick_eviction(&pool_entries)
    .ok_or(PoolError::Backpressure(self.retry_after))?;
```

`pick_eviction` returns `None` when every loaded model is busy (`in_flight > 0 || queued > 0`).
That refusal is correct and must stay — it is what stops a swap from preempting live work. The
defect is what happens next: we treat "cannot evict *right now*" as "cannot serve", when the
truthful answer is "servable after work that is already draining".

## Why the obvious fix does not work

"Wait for a victim to go idle" is necessary but not sufficient, and the reason is subtle enough
to be worth writing down.

`Gate::enter` **enqueues before it loads**:

```rust
self.manager.enqueue(real_id);          // A.queued += 1
... self.manager.ensure_loaded(...)     // then load/evict
```

So a newly arriving request for the *resident* model A raises `A.queued` before anything else
happens — and `pick_eviction` skips any model with `queued > 0`. Under steady traffic for A,
`A.queued` never returns to zero, so a waiter for B waits until its timeout and then 503s
anyway. We would have converted an immediate failure into a slow one, which is worse.

This is the starvation the issue anticipates, and it means the fairness rule is not a refinement
to add later — it is load-bearing for the fix to work at all.

## Design

Two cooperating pieces.

**1. A bounded wait for an evictable victim.** When `needs_eviction` holds and `pick_eviction`
returns `None`, register a *swap reservation* for the target model, then wait — re-checking
`pick_eviction` each time an entry goes idle — until a victim appears or the wait times out.
On timeout the answer is still `Backpressure`, unchanged. The waiter holds `load_lock`
throughout, which is correct: one swap at a time per device.

**2. A fairness hold, before `enqueue`.** While a reservation is outstanding, an arriving
request for a *different* model waits at the top of `Gate::enter` — **before** it enqueues —
until the reservation clears. That is what lets `A.queued` drain to zero. Requests already
admitted are untouched: the reservation never preempts in-flight work, it only stops new work
from jumping ahead of a waiter.

The reserving model itself is exempt, or it would block on its own reservation.

### Why this terminates

- A's in-flight requests finish on their own and call `release()`, which notifies the waiter.
- A's *new* requests are held before `enqueue`, so they cannot keep `A.queued` above zero.
- So A reaches `(in_flight 0, queued 0)`, `pick_eviction` picks it, and the swap proceeds.
- B is protected while it loads: `Gate::enter` enqueued B before calling `ensure_loaded`, so
  `B.queued > 0` for the whole load and a competing waiter cannot evict B before its caller is
  served. Every swap therefore serves at least one request, which is what makes the alternation
  progress rather than thrash.

### Bound

The wait is bounded by `queue_timeout`, per the issue: a swap is a cold load with an eviction in
front of it, and `queue_timeout` must already exceed a cold load. No new knob.

`queue_timeout` currently bounds only the semaphore acquire, not `ensure_loaded` (load time is
governed by `load_timeout`). Reusing it here extends its meaning to "how long a caller may wait
for its turn", which is what an operator setting it would expect it to mean.

## What must not regress

- A busy model is still never evicted. `pick_eviction` is unchanged.
- `503` is still the answer when the wait genuinely exceeds `queue_timeout` — the complaint is
  that it was returned *immediately*, not that it was returned.
- `queue_max` backpressure still short-circuits before any waiting.

## Test plan

Behaviour changes, so the existing expectation changes with it — deliberately, and separately
from the implementation:

1. `gate_protects_serving_model_from_eviction` currently asserts an *immediate* `Backpressure`.
   Under the new contract it must assert: waits, does not stop a busy model, and 503s only after
   the timeout. Its guarantee (never evict busy) is preserved; its timing claim is what changes.
2. New: a waiter for B blocks while A is held, and **succeeds** once A's slot is dropped.
3. New (the starvation case, and the one that fails without the fairness hold): a waiter for B,
   with a steady stream of new A requests arriving, still completes.
4. New: the wait is bounded — with A held open past `queue_timeout`, B gets `Backpressure`.
