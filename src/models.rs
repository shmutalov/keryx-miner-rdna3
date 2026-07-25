/// Registry of supported inference models.
///
/// model_id = sha2-256(primary_weight_file) = CIDv0_bytes[2..34].
/// Verifiable: decode the weight CID from base58btc, skip the 2-byte multihash prefix.
///
/// Uncensored five-tier lineup, active at `coin_age_verification_activation_daa()` (the H4
/// hardfork) — below that DAA this binary refuses to mine (`pom_tier_index` = None). Every
/// model is untied so the in-process llama engine hosts walk + inference in one resident copy:
///   --very-light  Qwen3-8B-ablit.   Q4_K_S (Alibaba)  — 6 GB+ (H5: replaced EXAONE-4.0-1.2B)
///   --light       Mistral-7B-v0.3  Q6_K   (Mistral)  — 8 GB
///   (default)     GLM-4-9B-0414    Q6_K   (Zhipu)    — 12 GB
///   --high        Qwen3.6-27B      Q4_K_M (Alibaba)  — 24 GB
///   --very-high   Kimi-Linear-48B  Q4_K_M (Moonshot) — 32 GB
///
/// All GGUF weights are pinned on the Keryx IPFS gateway; each
/// model_id = base58-decode(weight CID)[2..34].

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// GGUF quantized — LLaMA architecture (Mistral-7B). llama-served.
    Gguf,
    /// GGUF quantized — EXAONE 4 architecture. Retired as a PoM tier at H5 (it was tier 0); the
    /// format stays so an already-downloaded EXAONE GGUF still loads for inference.
    GgufExaone4,
    /// GGUF quantized — GLM 4 architecture (H4 tier 2). llama-served.
    GgufGlm4,
    /// GGUF quantized — Qwen3.5 hybrid-SSM architecture (H4 tier 3). llama-served.
    GgufQwen35,
    /// GGUF quantized — Kimi-Linear MoE architecture (H4 tier 4). llama-served.
    GgufKimiLinear,
}

#[derive(Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// 32-byte on-chain identifier embedded in AiRequest payloads.
    pub model_id: [u8; 32],
    pub format: ModelFormat,
    /// Empty for the whole H4 lineup: llama uses the tokenizer embedded in the GGUF.
    pub tokenizer_cid: &'static str,
    /// Single entry: the model.gguf CID.
    pub weight_cids: &'static [&'static str],
    /// Local directory name under `<exe_dir>/models/`.
    pub dir_name: &'static str,
    /// Minimum VRAM (MB) required to actually serve this model: weights +
    /// KV cache + workspace. Used by the OPoI capability gate so `ai:cap`
    /// never announces a model the miner cannot load. 0 = never gated.
    pub min_vram_mb: u64,
}

// ── H4 lineup ───────────────────────────────────────────────────
// Active at `crate::pom::coin_age_verification_activation_daa()` (the H4 hardfork). Every model is
// UNTIED so the in-process llama engine hosts walk + inference in one resident copy (zero-dup —
// no repack-materialized tensors, unlike the retired tied-embedding Gemma).
// `tokenizer_cid` is empty: llama uses the tokenizer embedded in the GGUF, no separate file.
// model_id bytes MUST equal the node's `params.rs` H4 constants (CIDv0[2..34] of the pinned GGUF).

/// H5 tier-0 model — Qwen3-8B-abliterated, which REPLACED EXAONE-4.0-1.2B at
/// `crate::pom::h5_activation_daa()`. The swap raises the tier-0 VRAM floor to ~6 GB, closing the
/// "any 2 GB card mines tier 0" gap: under H5 the mining table must be the real weights (see
/// `pom::transition_v2`), so tier 0 now demands a card that can actually serve the model.
/// model_id triple-checked against the node's `QWEN3_8B_ABLITERATED_MODEL_ID` and upstream's spec.
pub const QWEN3_8B_ABLITERATED: ModelSpec = ModelSpec {
    name: "qwen3-8b-abliterated",
    // CIDv0[2..34] of model.gguf — Qwen3-8B-abliterated Q4_K_S
    model_id: [
        0xd4, 0x2f, 0xa6, 0xee, 0x00, 0xe0, 0x7d, 0x49,
        0xb0, 0x46, 0x09, 0x0a, 0x56, 0xaf, 0x0e, 0x7b,
        0xd6, 0x10, 0x25, 0x93, 0x7c, 0x50, 0x2e, 0x2c,
        0x57, 0x4a, 0x72, 0x87, 0x4c, 0x35, 0x0d, 0x24,
    ],
    format: ModelFormat::GgufQwen35,
    // EMPTY, unlike upstream's spec which pins QmcuGkJvR343ry3b4jy7u5L9ior3ujas3yGAFMSyZdACb5:
    // this fork is llama-served end to end and reads the GGUF-embedded tokenizer (nothing in the
    // tree ever opens tokenizer.json — `slm.rs` only uses the CID to decide whether to fetch it).
    // A non-empty CID makes `slm::ensure_model` treat an already-complete model dir as incomplete,
    // which CLEARS the `.ok` flag and re-downloads the whole 4.8 GB GGUF. Non-consensus field
    // (model_id is the weight-file hash), so "" is safe and matches the rest of the lineup.
    tokenizer_cid: "",
    weight_cids: &["QmccwHVeZYVzEq6A5ofk76MxrnwzMnSjAVt9PaUQ7zfLXm"],
    dir_name: "Qwen3-8B-abliterated",
    // ~4.6 GB Q4_K_S weights + ~1.2 GB KV/workspace → fits a 6 GB card (measured 5,409 MiB @ ctx 4096).
    min_vram_mb: 6_000,
};

pub const MISTRAL_7B_V03: ModelSpec = ModelSpec {
    name: "mistral-7b-v0.3",
    // CIDv0[2..34] of model.gguf — Mistral-7B-Instruct-v0.3-abliterated Q6_K (mradermacher)
    model_id: [
        0x8c, 0x2f, 0xea, 0x60, 0x0f, 0x0e, 0xef, 0xe7,
        0x04, 0x87, 0x41, 0xa5, 0x11, 0x9c, 0xb7, 0xbe,
        0x30, 0x30, 0x37, 0xf5, 0x9f, 0xc0, 0x26, 0xe4,
        0x83, 0x82, 0x65, 0x8f, 0x23, 0x58, 0x1e, 0x0a,
    ],
    // llama architecture, but llama-served like the rest of the H4 lineup (no pinned tokenizer.json).
    format: ModelFormat::Gguf,
    tokenizer_cid: "",
    weight_cids: &["QmXmtATpJerCCcWWF515vAe5FanSvqJrD4L1ogZxDurQ3s"],
    dir_name: "Mistral-7B-v0.3",
    // ~5.9 GB Q6_K weights + ~1.5 GB KV/workspace. The Q6_K quant is deliberate VRAM gating:
    // it does NOT fit a 6 GB card, so 6 GB stays on tier 0 (EXAONE) and 8 GB serves tier 1.
    min_vram_mb: 8_000,
};

pub const GLM_4_9B_0414: ModelSpec = ModelSpec {
    name: "glm-4-9b-0414",
    // CIDv0[2..34] of model.gguf — GLM-4-9B-0414-abliterated Q6_K
    model_id: [
        0xfa, 0x2f, 0x13, 0xbe, 0x08, 0x50, 0xe2, 0x6c,
        0x5c, 0xe8, 0x6c, 0x7a, 0xc7, 0x9d, 0xa8, 0x5e,
        0x30, 0x0c, 0x1d, 0xa8, 0xb3, 0x29, 0x0f, 0x9a,
        0x18, 0xd4, 0x71, 0x05, 0xf1, 0xf2, 0x14, 0x0a,
    ],
    format: ModelFormat::GgufGlm4,
    tokenizer_cid: "",
    weight_cids: &["QmfBGGZumBR4XGFLLPjYozvhRSt3kXjrgsV3jXciCdAeM7"],
    dir_name: "GLM-4-9B-0414",
    // ~8.3 GB Q6_K weights + ~1.5 GB KV/workspace → 12 GB card.
    min_vram_mb: 12_000,
};

pub const QWEN3_6_27B: ModelSpec = ModelSpec {
    name: "qwen3.6-27b",
    // CIDv0[2..34] of model.gguf — Qwen3.6-27B-abliterated-v2 Q4_K_M (mradermacher)
    model_id: [
        0xb8, 0xbd, 0xc0, 0x1f, 0xa4, 0x07, 0xea, 0xb9,
        0x43, 0xe4, 0xfe, 0xfc, 0x80, 0x74, 0x83, 0xb3,
        0x9f, 0x81, 0x42, 0x78, 0x52, 0x56, 0x04, 0x9e,
        0x1f, 0x55, 0x96, 0x98, 0xa5, 0x28, 0x47, 0x46,
    ],
    format: ModelFormat::GgufQwen35,
    tokenizer_cid: "",
    weight_cids: &["QmamoYQGGAkBaqiWuNmwxeC9AQnt9F7sLyX57VoqbJWeUV"],
    dir_name: "Qwen3.6-27B",
    // ~16.5 GB Q4_K_M weights + ~2.5 GB KV/workspace → 24 GB card (7900 XTX-class; a 20 GB
    // 7900 XT is excluded by this gate, matching upstream's ladder).
    min_vram_mb: 24_000,
};

pub const KIMI_LINEAR_48B: ModelSpec = ModelSpec {
    name: "kimi-linear-48b",
    // CIDv0[2..34] of model.gguf — Kimi-Linear-48B-A3B-Instruct-abliterated Q4_K_M (mradermacher, i1)
    model_id: [
        0x3d, 0xc0, 0x93, 0x58, 0xad, 0x75, 0xc6, 0xef,
        0x0c, 0x9c, 0x86, 0xee, 0x4f, 0x47, 0xc4, 0xd6,
        0xac, 0xda, 0x96, 0x1f, 0xec, 0xbd, 0x0e, 0x4f,
        0x9c, 0xf5, 0x5e, 0x8f, 0x0f, 0xdf, 0xfd, 0xdb,
    ],
    format: ModelFormat::GgufKimiLinear,
    tokenizer_cid: "",
    weight_cids: &["QmSVhtoNrL8bWJXZuEXMMWqty8qHScQMRuacuoa9ujsYqp"],
    dir_name: "Kimi-Linear-48B",
    // ~29.7 GB Q4_K_M weights (MoE, 3B active) + KV/workspace → needs a 32 GB card.
    min_vram_mb: 30_000,
};

/// VRAM floor (MB) at which a card is still allowed to be ASSIGNED this model's PoM tier, as
/// opposed to `min_vram_mb` which states what serving it actually needs. The two differ only for
/// the H5 tier 0: Qwen3-8B loads in ~5,409 MiB @ ctx 4096, so the floor sits ~300 MiB below its
/// 6 GB `min_vram_mb` — a 6 GB card that reports slightly under 6000 (total_mem, driver-dependent)
/// keeps tier 0 — while staying ~300 MiB above the real load need for OOM margin. Its need is close
/// to `min_vram_mb`, so it cannot take a full 1 GB concession.
/// Mirrors upstream's `POM_TIER_LADDER` tier-0 floor of 5_700.
pub fn pom_assignment_floor_mb(spec: &ModelSpec) -> u64 {
    if spec.model_id == QWEN3_8B_ABLITERATED.model_id {
        5_700
    } else {
        spec.min_vram_mb
    }
}

/// Whether `model_id` is one of the Proof-of-Model tier models. DAA-independent —
/// used at startup to pick a mineable PoM model before any block DAA is known (the tier *index*
/// is then computed per block via `pom_tier_index`).
pub fn is_pom_model(model_id: &[u8; 32]) -> bool {
    *model_id == QWEN3_8B_ABLITERATED.model_id
        || *model_id == MISTRAL_7B_V03.model_id
        || *model_id == GLM_4_9B_0414.model_id
        || *model_id == QWEN3_6_27B.model_id
        || *model_id == KIMI_LINEAR_48B.model_id
}

/// Map a model_id to its Proof-of-Model tier index, matching the node's `POM_TIERS_H4` order.
/// H4 gate: below the flip this binary refuses to mine (None) — it never produces a
/// pre-H4-era block (the pre-H4 lineup was dropped with the H4-only refactor). MUST be
/// recomputed per block from that block's own DAA, never frozen at index-build time.
pub fn pom_tier_index(model_id: &[u8; 32], daa: u64) -> Option<u8> {
    if daa < crate::pom::coin_age_verification_activation_daa() {
        return None;
    }
    // Tier 0 is Qwen3-8B at/after H5. A Qwen3-8B tier-0 block below the gate is not a valid tier
    // (its R_T won't match the node's `POM_TIERS_H5`); the pre-H5 tier-0 model (EXAONE) is retired.
    if *model_id == QWEN3_8B_ABLITERATED.model_id {
        if daa >= crate::pom::h5_activation_daa() {
            Some(0)
        } else {
            None
        }
    } else if *model_id == MISTRAL_7B_V03.model_id {
        Some(1)
    } else if *model_id == GLM_4_9B_0414.model_id {
        Some(2)
    } else if *model_id == QWEN3_6_27B.model_id {
        Some(3)
    } else if *model_id == KIMI_LINEAR_48B.model_id {
        Some(4)
    } else {
        None
    }
}

/// Consensus-pinned PoM possession anchor: a model's canonical 32 B-chunk blake3 Merkle root `R_T`
/// and chunk count `N`, produced offline by the index builder.
pub struct PomAnchor {
    pub model_id: [u8; 32],
    pub root: [u8; 32],
    pub chunks: u64,
}

/// Per-model `(R_T, N)` anchors, copied VERBATIM from the node's `POM_TIERS_H4`
/// (`keryx-node consensus/core/src/config/params.rs`). The miner asserts its freshly-built
/// possession index matches the pinned `(root, N)` for the model it mines (see
/// `pom_gpu::ensure_installed_inner`), so a wrong-quant / corrupt / truncated GGUF is caught once
/// at index-build time — instead of silently producing PoM blocks every one of which the node
/// rejects with `BadWeightPath`. Keyed by model_id (not slice position) so the check stays correct
/// even if the node later reorders tiers.
pub const POM_ANCHORS: &[PomAnchor] = &[
    PomAnchor {
        // Tier 0 @ H5 — Qwen3-8B-abliterated Q4_K_S. Values copied from the node's
        // `POM_TIERS_H5[0]` (consensus/core/src/config/params.rs); the retired EXAONE anchor
        // (root cc8b25c4…, 28,943,588 chunks) is gone with the model.
        model_id: QWEN3_8B_ABLITERATED.model_id,
        root: [
            0xa1, 0xcb, 0xff, 0xfa, 0xae, 0xb9, 0x71, 0xcb, 0x29, 0x7b, 0x7e, 0x01, 0xff, 0x41, 0x09, 0x72,
            0x3e, 0x43, 0x97, 0x41, 0xcd, 0x42, 0x68, 0x22, 0x5f, 0x0c, 0x30, 0xa3, 0x33, 0xe6, 0x9a, 0x68,
        ],
        chunks: 149_876_736,
    },
    PomAnchor {
        model_id: MISTRAL_7B_V03.model_id,
        root: [
            0xd7, 0x6a, 0xcb, 0xbe, 0x8c, 0x24, 0x29, 0x81, 0x6c, 0x02, 0xa4, 0xdb, 0xd9, 0xf2, 0x09, 0xa0,
            0x87, 0x85, 0xef, 0x97, 0x5c, 0xd1, 0x38, 0xf5, 0x18, 0x22, 0x76, 0x12, 0x0b, 0xa2, 0x0e, 0xc5,
        ],
        chunks: 185_827_840,
    },
    PomAnchor {
        model_id: GLM_4_9B_0414.model_id,
        root: [
            0x1b, 0xa8, 0xb8, 0xb1, 0x34, 0x41, 0x03, 0xfa, 0xa0, 0xa7, 0x47, 0x89, 0xd9, 0x39, 0xc3, 0x3c,
            0x23, 0xba, 0x5c, 0x3c, 0x41, 0xbb, 0x1a, 0x89, 0x5a, 0xb6, 0xe8, 0xbf, 0xec, 0xb0, 0x78, 0x7d,
        ],
        chunks: 258_040_832,
    },
    PomAnchor {
        model_id: QWEN3_6_27B.model_id,
        root: [
            0x85, 0x23, 0xf4, 0x14, 0x8d, 0x22, 0xc7, 0x71, 0x3b, 0xfc, 0x11, 0x32, 0xb4, 0xaf, 0x3d, 0x4b,
            0x97, 0x61, 0xa2, 0x03, 0xfb, 0x33, 0xf1, 0x8e, 0xe7, 0x55, 0x67, 0xbd, 0xee, 0x51, 0x2b, 0x0a,
        ],
        chunks: 516_762_688,
    },
    PomAnchor {
        model_id: KIMI_LINEAR_48B.model_id,
        root: [
            0x95, 0x74, 0x71, 0x0f, 0xfa, 0xb6, 0x78, 0xf0, 0x68, 0xb4, 0xe6, 0x5a, 0xbe, 0x72, 0x40, 0x86,
            0x2d, 0xa1, 0x5b, 0xb1, 0x6e, 0xa8, 0x2f, 0xd1, 0x62, 0xa9, 0x35, 0x1a, 0x10, 0x51, 0x99, 0x59,
        ],
        chunks: 927_994_064,
    },
];

/// The consensus-pinned PoM anchor for `model_id`, if it is a known PoM tier model.
pub fn pinned_pom_anchor(model_id: &[u8; 32]) -> Option<&'static PomAnchor> {
    POM_ANCHORS.iter().find(|a| &a.model_id == model_id)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    VeryLight,
    Light,
    Default,
    High,
    VeryHigh,
}

/// The single model a hardware tier mines AND serves — one flag = one model. A PoM GPU is
/// bound to its tier (serving a lower tier would mean unloading the mined model and pausing
/// mining); multi-tier coverage is a network property (different miners per tier), not a
/// per-GPU one.
pub fn spec_for_tier(tier: Tier) -> &'static ModelSpec {
    match tier {
        Tier::VeryLight => &QWEN3_8B_ABLITERATED,
        Tier::Light => &MISTRAL_7B_V03,
        Tier::Default => &GLM_4_9B_0414,
        Tier::High => &QWEN3_6_27B,
        Tier::VeryHigh => &KIMI_LINEAR_48B,
    }
}

/// [`spec_for_tier`] as a one-element static slice — the shape the staging/announce path
/// (`init_supported`/`prefetch_models`) consumes.
pub fn specs_for_tier(tier: Tier) -> &'static [&'static ModelSpec] {
    match tier {
        Tier::VeryLight => &[&QWEN3_8B_ABLITERATED],
        Tier::Light => &[&MISTRAL_7B_V03],
        Tier::Default => &[&GLM_4_9B_0414],
        Tier::High => &[&QWEN3_6_27B],
        Tier::VeryHigh => &[&KIMI_LINEAR_48B],
    }
}

/// Resolves a model name/id.
pub const REGISTRY: &[&ModelSpec] = &[
    &QWEN3_8B_ABLITERATED,
    &MISTRAL_7B_V03,
    &GLM_4_9B_0414,
    &QWEN3_6_27B,
    &KIMI_LINEAR_48B,
];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().copied().find(|m| m.name == name)
}

pub fn available_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|m| m.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The gates are runtime fns now (`--testnet`), so these are `fn`s rather than `const`s.
    // The H4 gate is consensus-critical: the tier index emitted in each PoM proof MUST match the
    // node's per-block tier table per the block's own DAA, and MUST be None below the flip.
    fn pre_h4() -> u64 {
        crate::pom::coin_age_verification_activation_daa() - 1
    }
    fn at_h5() -> u64 {
        crate::pom::h5_activation_daa()
    }

    #[test]
    fn pom_tier_index_refuses_below_h4() {
        for m in REGISTRY {
            assert_eq!(pom_tier_index(&m.model_id, pre_h4()), None, "{} must not mine pre-H4", m.name);
        }
    }

    #[test]
    fn pom_tier_index_at_h5_matches_node_order() {
        assert_eq!(pom_tier_index(&QWEN3_8B_ABLITERATED.model_id, at_h5()), Some(0));
        assert_eq!(pom_tier_index(&MISTRAL_7B_V03.model_id, at_h5()), Some(1));
        assert_eq!(pom_tier_index(&GLM_4_9B_0414.model_id, at_h5()), Some(2));
        assert_eq!(pom_tier_index(&QWEN3_6_27B.model_id, at_h5()), Some(3));
        assert_eq!(pom_tier_index(&KIMI_LINEAR_48B.model_id, at_h5()), Some(4));
    }

    /// Tier 0 is era-gated: Qwen3-8B is only a valid tier 0 at/after H5. Below the gate its R_T
    /// does not match any node tier table, so claiming it would earn a BadWeightPath rejection.
    #[test]
    fn qwen3_8b_is_not_a_tier_below_h5() {
        let below_h5 = crate::pom::h5_activation_daa() - 1;
        assert!(below_h5 >= crate::pom::coin_age_verification_activation_daa(), "H5 must sit above H4");
        assert_eq!(pom_tier_index(&QWEN3_8B_ABLITERATED.model_id, below_h5), None);
        // The other tiers are era-stable across the H5 crossing.
        assert_eq!(pom_tier_index(&MISTRAL_7B_V03.model_id, below_h5), Some(1));
    }

    #[test]
    fn every_pom_model_has_a_pinned_anchor() {
        // Tier order: Qwen3-8B (H5 tier 0), Mistral-7B, GLM-4-9B, Qwen3.6-27B, Kimi-Linear-48B.
        let expected_chunks = [149_876_736u64, 185_827_840, 258_040_832, 516_762_688, 927_994_064];
        for (m, chunks) in REGISTRY.iter().zip(expected_chunks) {
            let a = pinned_pom_anchor(&m.model_id).unwrap_or_else(|| panic!("{} anchor missing", m.name));
            assert_eq!(a.chunks, chunks, "{} chunk count", m.name);
            assert!(is_pom_model(&m.model_id));
        }
    }
}
