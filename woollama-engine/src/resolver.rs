//! Pure model-resolution + eviction decision logic (server-free).
//!
//! `resolve` turns a virtual/bare model id into a concrete device model id using a
//! snapshot of what's loaded; `needs_eviction`/`pick_eviction` decide, from a
//! snapshot of per-model runtime state, whether and which idle model to unload to
//! make room. No I/O, no async, no server dependencies — unit-testable in isolation
//! and a direct port of `woollama.resolver` (Python oracle).

use std::collections::{BTreeMap, HashSet};

/// Read-only snapshot of one loaded model's runtime state, for eviction.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolEntry {
    pub model_id: String,
    pub in_flight: u32,
    pub queued: u32,
    pub last_used: f64,
}

/// A virtual model could not be resolved (e.g. `default` with nothing loaded
/// and no configured fallback). The router maps this to a clear client error.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveError(pub String);

/// Resolve the id after `provider/` to a concrete device model id.
///
/// - `default` -> the currently-loaded model (`loaded[0]`, MRU first); if none
///   loaded, the configured fallback `default`; else raise `ResolveError`.
/// - a `bare` present in the `virtual_models` alias map -> its real id.
/// - anything else -> `bare` unchanged (real-id passthrough, today's behavior).
pub fn resolve(
    bare: &str,
    virtual_models: &BTreeMap<String, String>,
    loaded: &[String],
    default: Option<&str>,
) -> Result<String, ResolveError> {
    if bare == "default" {
        if let Some(first) = loaded.first() {
            return Ok(first.clone());
        }
        if let Some(default) = default {
            return Ok(default.to_string());
        }
        return Err(ResolveError(
            "model 'default' requested but no model is loaded and no \
             'virtual.default' fallback is configured for this inferencer"
                .to_string(),
        ));
    }
    if let Some(target) = virtual_models.get(bare) {
        return Ok(target.clone());
    }
    Ok(bare.to_string())
}

/// True iff a cap is set, is reached, and `target` is not already loaded.
pub fn needs_eviction(loaded: &HashSet<String>, target: &str, pool_max: Option<u32>) -> bool {
    let Some(pool_max) = pool_max else {
        return false;
    };
    if pool_max == 0 {
        return false;
    }
    if loaded.contains(target) {
        return false;
    }
    loaded.len() as u32 >= pool_max
}

/// The LRU model among idle entries (no in-flight, empty queue), or `None` if
/// every loaded model is busy (never evict a serving/queued model).
pub fn pick_eviction(entries: &[PoolEntry]) -> Option<String> {
    entries
        .iter()
        .filter(|e| e.in_flight == 0 && e.queued == 0)
        .min_by(|a, b| a.last_used.partial_cmp(&b.last_used).unwrap())
        .map(|e| e.model_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn strs(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn set(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn resolve_real_id_passthrough() {
        assert_eq!(
            resolve("Qwen/Coder", &BTreeMap::new(), &[], None).unwrap(),
            "Qwen/Coder"
        );
    }

    #[test]
    fn resolve_default_prefers_loaded() {
        let virtual_models = map(&[("default", "Cfg")]);
        let loaded = strs(&["Loaded/A", "Loaded/B"]);
        assert_eq!(
            resolve("default", &virtual_models, &loaded, Some("Cfg")).unwrap(),
            "Loaded/A"
        );
    }

    #[test]
    fn resolve_default_falls_back_to_config_when_none_loaded() {
        let virtual_models = map(&[("default", "Cfg")]);
        assert_eq!(
            resolve("default", &virtual_models, &[], Some("Cfg")).unwrap(),
            "Cfg"
        );
    }

    #[test]
    fn resolve_default_no_loaded_no_config_raises() {
        let result = resolve("default", &BTreeMap::new(), &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_alias_maps_to_real_id() {
        let virtual_models = map(&[("coder", "Qwen/Coder")]);
        assert_eq!(
            resolve("coder", &virtual_models, &[], None).unwrap(),
            "Qwen/Coder"
        );
    }

    #[test]
    fn resolve_unknown_alias_returns_itself() {
        let virtual_models = map(&[("coder", "Qwen/Coder")]);
        let loaded = strs(&["X"]);
        assert_eq!(
            resolve("mystery", &virtual_models, &loaded, None).unwrap(),
            "mystery"
        );
    }

    #[test]
    fn needs_eviction_only_when_capped_full_and_target_absent() {
        assert!(needs_eviction(&set(&["a", "b"]), "c", Some(2))); // capped, full, absent
        assert!(!needs_eviction(&set(&["a", "b"]), "a", Some(2))); // already loaded
        assert!(!needs_eviction(&set(&["a"]), "c", Some(2))); // room
        assert!(!needs_eviction(&set(&["a", "b"]), "c", None)); // no cap
        assert!(!needs_eviction(&set(&["a", "b"]), "c", Some(0))); // cap of zero
    }

    #[test]
    fn pick_eviction_lru_idle() {
        let entries = vec![
            PoolEntry {
                model_id: "old".to_string(),
                in_flight: 0,
                queued: 0,
                last_used: 1.0,
            },
            PoolEntry {
                model_id: "new".to_string(),
                in_flight: 0,
                queued: 0,
                last_used: 9.0,
            },
        ];
        assert_eq!(pick_eviction(&entries), Some("old".to_string()));
    }

    #[test]
    fn pick_eviction_never_picks_busy() {
        let entries = vec![
            PoolEntry {
                model_id: "serving".to_string(),
                in_flight: 1,
                queued: 0,
                last_used: 1.0,
            },
            PoolEntry {
                model_id: "queued".to_string(),
                in_flight: 0,
                queued: 2,
                last_used: 2.0,
            },
        ];
        assert_eq!(pick_eviction(&entries), None);
    }

    #[test]
    fn pick_eviction_skips_busy_returns_idle() {
        let entries = vec![
            PoolEntry {
                model_id: "serving".to_string(),
                in_flight: 1,
                queued: 0,
                last_used: 1.0,
            },
            PoolEntry {
                model_id: "idle".to_string(),
                in_flight: 0,
                queued: 0,
                last_used: 5.0,
            },
        ];
        assert_eq!(pick_eviction(&entries), Some("idle".to_string()));
    }
}
