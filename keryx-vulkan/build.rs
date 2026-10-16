//! Compile every `shaders/*.comp` GLSL compute shader to SPIR-V (`$OUT_DIR/<name>.spv`) with
//! `glslc` (from the Vulkan SDK). The `.spv` is `include_bytes!`'d into the crate, so the built
//! miner carries its shaders inline and needs no SDK at runtime — only the vulkan-1 loader.

use std::path::PathBuf;
use std::process::Command;

fn find_glslc() -> PathBuf {
    // Prefer an explicit override, then the Vulkan SDK's Bin dir, then PATH.
    if let Ok(p) = std::env::var("GLSLC") {
        return PathBuf::from(p);
    }
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let exe = if cfg!(windows) { "glslc.exe" } else { "glslc" };
        let candidate = PathBuf::from(sdk).join("Bin").join(exe);
        if candidate.exists() {
            return candidate;
        }
    }
    PathBuf::from("glslc") // assume on PATH
}

/// Extra builds of a shader under preprocessor defines: (source stem, output stem, defines).
const VARIANTS: &[(&str, &str, &[&str])] = &[
    // The PoM v4 walk with its transition on the matrix cores (VK_KHR_cooperative_matrix).
    ("pom_walk_v4", "pom_walk_v4_cm", &["-DPOM_V4_CM"]),
];

fn compile(glslc: &PathBuf, src: &std::path::Path, spv: &std::path::Path, defines: &[&str]) {
    let status = Command::new(glslc)
        .args(["-O", "--target-env=vulkan1.3"])
        .args(defines)
        .arg(src)
        .arg("-o")
        .arg(spv)
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to run glslc ({}): {e}. Install the Vulkan SDK or set GLSLC/VULKAN_SDK.", glslc.display())
        });
    assert!(status.success(), "glslc failed to compile {} {defines:?}", src.display());
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let glslc = find_glslc();

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let entries = std::fs::read_dir(&shader_dir).expect("read shaders/ dir");
    for entry in entries {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("comp") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_str().unwrap();
        println!("cargo:rerun-if-changed={}", path.display());
        compile(&glslc, &path, &out_dir.join(format!("{stem}.spv")), &[]);
        for (_, out_stem, defines) in VARIANTS.iter().filter(|(src, ..)| *src == stem) {
            compile(&glslc, &path, &out_dir.join(format!("{out_stem}.spv")), defines);
        }
    }
}
