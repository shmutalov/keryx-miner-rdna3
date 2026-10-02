//! PoM v4 (D=32 re-walk) — wire structs, constants, host proof-build. Byte-exact mirror of the
//! node's `consensus/core/src/pom_v4.rs`. Field order and salts MUST stay bit-identical.

use crate::pom::{blake, merkle_root, mix64, verify_merkle, WeightIndex};
use crate::pom_v3::{dot_i8, fold64, rho8, rho_tweak, snippet_fold};
use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};

pub const POM_V4_D: usize = 32;
pub const POM_V4_K: usize = 256;
pub const POM_V4_CHUNK_BYTES: usize = 32;
pub const POM_V4_TILE_BYTES: usize = POM_V4_D * POM_V4_D; // 1 KB
pub const POM_V4_TILE_CHUNKS: u64 = (POM_V4_TILE_BYTES / POM_V4_CHUNK_BYTES) as u64;
pub const POM_V4_SNIPPET_BYTES: usize = 32;
pub const POM_V4_TILE_SUBTREE_DEPTH: u32 = 5; // log2(POM_V4_TILE_CHUNKS)

/// sha256("keryx-v4-s0-row-salt")
pub const POM_V4_S0_ROW_SALT: u64 = 0x03421325594C3C51;
/// sha256("keryx-v4-offset-first-salt")
pub const POM_V4_OFFSET_FIRST_SALT: u64 = 0x6D1CCF96AC4D76F9;
/// sha256("keryx-v4-offset-step-salt")
pub const POM_V4_OFFSET_STEP_SALT: u64 = 0x89050E78D34609EF;

/// Merkle range proof for one tile (path from the tile's aligned subtree root up to R_T).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomV4RangeProof {
    pub path: Vec<[u8; 32]>,
}

/// v4 walk witness — mirror of the node's `PomProofV4` (borsh field order).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofV4 {
    pub tier: u8,
    pub tiles: Vec<Vec<u8>>,
    pub merkle: Vec<PomV4RangeProof>,
}

#[inline]
pub fn v4_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V4_OFFSET_FIRST_SALT) % n_tiles
}

#[inline]
pub fn v4_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V4_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

pub fn v4_initial_state(seed: u64) -> Vec<u8> {
    let mut s = vec![0u8; POM_V4_D * POM_V4_D];
    for r in 0..POM_V4_D {
        let mut h = mix64(seed ^ POM_V4_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V4_D / 4 {
            h = mix64(h);
            s[r * POM_V4_D + k4 * 4..r * POM_V4_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

pub fn v4_transition(state: &[u8], tile: &[u8], step: u32) -> Vec<u8> {
    let mut next = vec![0u8; POM_V4_D * POM_V4_D];
    for x in 0..POM_V4_D {
        let row = &state[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            next[x * POM_V4_D + j] = rho8(dot_i8(row, col), rho_tweak(step, x as u32, j as u32));
        }
    }
    next
}

fn v4_state_leaves(state: &[u8]) -> Vec<[u8; 32]> {
    (0..POM_V4_D).map(|r| blake(&state[r * POM_V4_D..(r + 1) * POM_V4_D])).collect()
}

pub fn v4_state_root(state: &[u8]) -> [u8; 32] {
    merkle_root(&v4_state_leaves(state))
}

fn v4_tile_subtree_root(tile: &[u8]) -> [u8; 32] {
    let leaves: Vec<[u8; 32]> = tile.chunks(POM_V4_CHUNK_BYTES).map(blake).collect();
    merkle_root(&leaves)
}

/// The v4 walk's `final_state` for `seed` over `index`, without collecting the witness — the cheap
/// target pre-check before [`build_proof_v4`] (no Merkle paths are computed).
pub fn walk_final_v4(seed: u64, index: &WeightIndex) -> Result<u64> {
    let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small for the v4 walk"));
    }
    let mut state = v4_initial_state(seed);
    let mut off = v4_first_offset(seed, n_tiles);
    let mut tile = Vec::with_capacity(POM_V4_TILE_BYTES);
    for step in 1..=POM_V4_K as u64 {
        tile.clear();
        for c in 0..POM_V4_TILE_CHUNKS {
            tile.extend_from_slice(&index.read_chunk_bytes(off * POM_V4_TILE_CHUNKS + c));
        }
        state = v4_transition(&state, &tile, step as u32);
        if step < POM_V4_K as u64 {
            let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v4_next_offset(seed, step, &snippet, n_tiles);
        }
    }
    Ok(fold64(&v4_state_root(&state)))
}

/// Re-walk `seed` reading tiles from `index`, returning the proof and the derived `final_state`.
pub fn build_proof_v4(tier: u8, seed: u64, index: &WeightIndex) -> Result<(PomProofV4, u64)> {
    let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small for the v4 walk"));
    }
    let mut state = v4_initial_state(seed);
    let mut off = v4_first_offset(seed, n_tiles);
    let mut tiles = Vec::with_capacity(POM_V4_K);
    let mut merkle = Vec::with_capacity(POM_V4_K);
    for step in 1..=POM_V4_K as u64 {
        let mut tile = Vec::with_capacity(POM_V4_TILE_BYTES);
        for c in 0..POM_V4_TILE_CHUNKS {
            tile.extend_from_slice(&index.read_chunk_bytes(off * POM_V4_TILE_CHUNKS + c));
        }
        let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
        let path = index.merkle_path(off * POM_V4_TILE_CHUNKS)[POM_V4_TILE_SUBTREE_DEPTH as usize..].to_vec();
        merkle.push(PomV4RangeProof { path });
        state = v4_transition(&state, &tile, step as u32);
        tiles.push(tile);
        if step < POM_V4_K as u64 {
            off = v4_next_offset(seed, step, &snippet, n_tiles);
        }
    }
    let final_state = fold64(&v4_state_root(&state));
    Ok((PomProofV4 { tier, tiles, merkle }, final_state))
}

/// Pre-submit self-check: re-walk the proof against `r_t` and return the derived `final_state`.
pub fn verify_proof_v4(seed: u64, proof: &PomProofV4, r_t: &[u8; 32], n_chunks: u64) -> Result<u64> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(anyhow!("v4 proof wrong shape"));
    }
    let n_tiles = n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small"));
    }
    let mut state = v4_initial_state(seed);
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K {
        let tile = &proof.tiles[step - 1];
        if tile.len() != POM_V4_TILE_BYTES {
            return Err(anyhow!("v4 tile wrong shape"));
        }
        if !verify_merkle(v4_tile_subtree_root(tile), off, &proof.merkle[step - 1].path, r_t) {
            return Err(anyhow!("v4 tile fails range proof at step {step}"));
        }
        state = v4_transition(&state, tile, step as u32);
        if step < POM_V4_K {
            let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(fold64(&v4_state_root(&state)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pom::{index_from_ram, pom_block_seed_h10, pom_block_seed_v4, pom_pow_value, PomProof};
    use crate::pom_v3::lockstep_blob;

    // Cross-implementation vectors over `pom_v3::lockstep_blob()` (32_773 chunks = 1024 tiles),
    // produced by an independent pure-Python implementation of the v4 walk, the H10 seed and the
    // R_T Merkle tree (itself checked against every vector pinned upstream).
    const PPH: [u8; 32] = [7u8; 32];
    const TS: u64 = 1_788_000_000_000;
    const NONCE: u64 = 0x1234_5678_9ABC_DEF0;

    #[test]
    fn v4_walk_matches_the_reference_vectors() {
        let idx = index_from_ram(lockstep_blob());
        let n_tiles = idx.n_chunks / POM_V4_TILE_CHUNKS;
        assert_eq!(n_tiles, 1024);

        // Raw-seed walk.
        let raw_seed = 0x0F1E_2D3C_4B5A_6978u64;
        assert_eq!(v4_first_offset(raw_seed, n_tiles), 887);
        assert_eq!(walk_final_v4(raw_seed, &idx).unwrap(), 0xd8e6_25e8_d975_a7c1);

        // End to end: H10 seed -> walk -> H3 pow fold.
        let seed = pom_block_seed_h10(&PPH, TS, NONCE);
        assert_eq!(seed, 0x11f6_89ad_29bb_12c4);
        assert_eq!(pom_block_seed_v4(&PPH, TS, NONCE), 0x01d9_18dd_f95a_b1e6);
        assert_eq!(v4_first_offset(seed, n_tiles), 240);
        let final_state = walk_final_v4(seed, &idx).unwrap();
        assert_eq!(final_state, 0x838a_4d97_dd79_f2c2);
        assert_eq!(
            hex::encode(pom_pow_value(final_state, &PPH, true)),
            "cf2fd53f34984a3d3cc1ddc7d693fcf7f2212a056351675740441811a8ae8ba0"
        );
    }

    #[test]
    fn v4_proof_self_verifies_and_encodes_era_exact() {
        let idx = index_from_ram(lockstep_blob());
        let seed = pom_block_seed_h10(&PPH, TS, NONCE);
        let (v4, final_state) = build_proof_v4(2, seed, &idx).unwrap();
        assert_eq!(final_state, 0x838a_4d97_dd79_f2c2);
        assert_eq!(verify_proof_v4(seed, &v4, &idx.r_t, idx.n_chunks).unwrap(), final_state);

        // A tampered tile must fail its range proof.
        let mut bad = v4.clone();
        bad.tiles[17][3] ^= 1;
        assert!(verify_proof_v4(seed, &bad, &idx.r_t, idx.n_chunks).is_err());

        let proof = PomProof {
            tier: 2,
            trace_root: [0u8; 32],
            pow_value: pom_pow_value(final_state, &PPH, true),
            final_state,
            initial_trace_path: vec![],
            final_trace_path: vec![],
            openings: vec![],
            steps_v2: None,
            v3: None,
            v4: Some(v4),
        };
        let wire = proof.to_wire_bytes();
        // 264_289 + 8_192 * L bytes with an 11-level tile path (ceil(log2(1025)) = 11).
        assert_eq!(wire.len(), 354_401);
        // Legacy prefix canonically empty, then the Option tags: steps_v2 None, v3 None, v4 Some.
        assert_eq!(&wire[73..85], &[0u8; 12]);
        assert_eq!(&wire[85..88], &[0, 0, 1]);
        let back = PomProof::from_wire_bytes(&wire).unwrap();
        assert!(back.v3.is_none() && back.v4.is_some());
        assert_eq!(back.final_state, final_state);
    }
}
