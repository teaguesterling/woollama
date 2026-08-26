# Protecting in-flight work from device-forced swaps (#47)

## The gap

woollama's eviction protection assumes **woollama is the only party that evicts**. `pick_eviction`
refuses any model with `in_flight > 0` or `queued > 0`, and that works — verified on hardware,
5/5 holder requests served through a swap and a swap back.

It works only when we are the one doing the evicting. On the deployment that matters it is not:

- `pool_max` is **deliberately unset**, because it counts *models* and the device counts
  *capacity*. Six models resident is a working state there (55/55/32/2/1%), so `pool_max = 1`
  forbids what the hardware supports and `2` permits what it cannot hold. There is no correct
  integer.
- With `pool_max` unset, `needs_eviction` is always false. We never evict. We simply issue
  `start`, and **the device makes room by killing something**.

Measured: one in-flight request dies per swap, deterministically, every run. A logging proxy
recorded **zero `stop` calls** across a full contention cycle — the protection did not fail, it
never applied.

## The lever is *when* we issue `start`, not *whether* we evict

The proxy log showed zero `stop`s and two `start`s across a swap cycle. Nothing else initiates an
eviction on that device. So although we cannot *decline* a device-forced eviction, we have
complete control over **the moment it happens** — which is a stronger position than the
`pool_max` path gives us, where we chose a victim; here we choose a time.

## Scope of this change — deliberately smaller than the issue

The issue sketches a full capacity model: read the static `npu_usage` table from the catalogue,
compute `sum(resident) + target > 100` locally, and use `unload_candidate` only as a pre-flight.
That is the right end state and it is **not** what this change does.

This change does the minimum that closes the actual defect:

> Immediately before issuing `start`, ask the device whether that `start` will force an eviction
> and of what. If it will, and the named victim has work we know about, wait for that work to
> drain — bounded by `queue_timeout` — then issue the `start`.

Why stop there:

- It delivers the entire user-visible benefit (no request killed by our own load) without a new
  capacity model to get wrong.
- It costs **one call per load**, not per request. Measured 65–268 ms, mean ~192 ms, against a
  cold load of 25–30 s. Free where it happens; unacceptable if it were on the request path, which
  is exactly why the local-arithmetic optimisation exists as a later step.
- It works with `pool_max` unset, which is the configuration the defect lives in. It does not
  require an operator to supply a number that has no correct value.

The local `npu_usage` arithmetic is a follow-up whose purpose is to *remove the per-load call and
make `pool_max` unnecessary*, not to fix this. Sequencing it second keeps the risky part (touching
the eviction path) separate from the cheap part (not asking the device twice).

## Design

`DeviceBackend` gains one method:

```rust
/// Will loading `id` force the device to evict something, and if so what?
///
/// `Ok(None)` means "no eviction required, or this backend cannot say" — the two are
/// deliberately the same answer, because a backend that cannot say must not be treated as
/// having said "no". Callers get no new protection; they are exactly where they are today.
async fn unload_candidate(&self, id: &str) -> Result<Option<String>, PoolError> { Ok(None) }
```

Default `Ok(None)`, so Ollama and every config-defined REST protocol are unaffected. Only the
`device` preset overrides it.

In `ensure_loaded`, immediately before `self.backend.load(real_id)`:

1. Ask `unload_candidate(real_id)`.
2. `None` ⇒ load now, as today.
3. `Some(victim)` ⇒ if our bookkeeping shows the victim idle, load now. Otherwise wait on the
   existing swap-wakeup epoch until it drains or the deadline passes, then load **regardless**.

Step 3's "load regardless" is deliberate. The deadline is a bound on *politeness*, not a veto: if
the victim never drains we still serve the caller, because refusing would trade a request we might
have lost for one we definitely lose.

## What this cannot do, and must say so

**We can only drain work we know about.** Another consumer's in-flight request on the victim —
one not routed through this woollamad — is invisible to us, and the device will kill it. That is
a property of the hardware and belongs in the documentation, not in a promise.

The device also will not protect a busy model on its own: measured with a holder issuing a request
every 0.5 s, continuously busy, and the device evicted it mid-stream anyway.

## Explicitly not used

`active_request_count` is **offset, not stale** — a freshly loaded instance that has served
nothing reads 1, or 5, and the floor is per-instance. `count > 0` is the wrong predicate and
`count > baseline` needs re-sampling after every swap. `/npu/status` occupants has a true zero and
needs no calibration. Neither is needed here: we gate on **our own** in-flight bookkeeping, which
we already maintain and which has a real zero.

## Test plan

1. A backend reporting no eviction required ⇒ load proceeds immediately, no wait.
2. A backend reporting a victim that is **idle** ⇒ load proceeds immediately.
3. A backend reporting a victim that is **busy** ⇒ the load waits, and happens once the victim's
   slot drops. This is the defect.
4. A victim that never drains ⇒ the load still happens after the deadline, and the caller is
   served rather than refused.
5. A backend that does not implement it ⇒ behaviour identical to today.
