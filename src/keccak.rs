//! Keccak-f[1600] permutation for the lib crate (the H10 PoM walk seed). Same backend selection as
//! the binary's legacy PowHash (`src/pow/keccak.rs`): the vendored x86-64 asm where it is linked,
//! the `keccak` crate elsewhere.

#[cfg(any(not(target_arch = "x86_64"), feature = "no-asm", target_os = "windows"))]
pub fn f1600(state: &mut [u64; 25]) {
    keccak::f1600(state);
}

#[cfg(all(target_arch = "x86_64", not(feature = "no-asm"), not(target_os = "windows")))]
pub fn f1600(state: &mut [u64; 25]) {
    extern "C" {
        fn KeccakF1600(state: &mut [u64; 25]);
    }
    unsafe { KeccakF1600(state) }
}
