// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

//! rav1e is an [AV1] video encoder. It is designed to eventually cover all use
//! cases, though in its current form it is most suitable for cases where
//! libaom (the reference encoder) is too slow.
//!
//! ## Features
//!
//! * Intra and inter frames
//! * 64x64 superblocks
//! * 4x4 to 64x64 RDO-selected square and 2:1/1:2 rectangular blocks
//! * DC, H, V, Paeth, smooth, and a subset of directional prediction modes
//! * DCT, (FLIP-)ADST and identity transforms (up to 64x64, 16x16 and 32x32
//!   respectively)
//! * 8-, 10- and 12-bit depth color
//! * 4:2:0 (full support), 4:2:2 and 4:4:4 (limited) chroma sampling
//! * Variable speed settings
//! * Near real-time encoding at high speed levels
//!
//! ## Usage
//!
//! Encoding is done through the [`Context`] struct. Examples on
//! [`Context::receive_packet`] show how to create a [`Context`], send frames
//! into it and receive packets of encoded data.
//!
//! [AV1]: https://aomediacodec.github.io/av1-spec/av1-spec.pdf
//! [`Context`]: struct.Context.html
//! [`Context::receive_packet`]: struct.Context.html#method.receive_packet

#![allow(missing_abi)]
#![allow(unused_unsafe)]

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

pub use crate::api::color;
pub use crate::api::{
  Config, Context, EncoderConfig, EncoderStatus, InvalidConfig, Packet,
};
use crate::encoder::*;
pub use crate::frame::Frame;
pub use crate::util::{CastFromPrimitive, Pixel, PixelType};

// phasm-stego streaming-session lookahead (Option B Phase P1, #232).
// Free-function counterpart to ContextInner::compute_block_importances
// for the encode_gop_with_phasm_tee path. See module docstring + the
// implementation plan at
// docs/design/video/av1/av1-stealth-lookahead-plan-2026-06-29.md.
mod phasm_stego_lookahead;

// phasm-stego (W3.8.3, Option 3 minimal-API):
// Re-exports of internal types needed by phasm-core to call
// `crate::encoder::encode_tile::<WriterRecorder>` directly. See
// `phasm-av1/docs/design/video/av1/rav1e-hook-sites.md` § 3.
//
// TODO(v0.4+ refactor): These re-exports should be replaced by a
// cleaner Option 1 API (encode_frame_for_pass1 + encode_frame_wrap_obu)
// that takes the same args as the public Context::receive_packet
// surface, so phasm-core doesn't depend on these internal types.
// See rav1e-hook-sites.md § 3.2 + § 9 Q-OPT1.
pub mod phasm_stego {
  // P1 lookahead refinement (#232 — av1-stealth-lookahead-plan-2026-06-29.md):
  pub use crate::phasm_stego_lookahead::{
    compute_distortion_scales_for_window, LookaheadWindowFrame,
  };

  pub use crate::api::PhasmInterConfig as InterConfig;
  pub use crate::context::FrameBlocks;
  pub use crate::ec::{
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB, PHASM_TAG_OTHER,
    // W3.10.3: WriterTee combined Encoder+Recorder backend.
    WriterTee,
    // W3.10.4: recorder data types returned from
    // encode_frame_with_phasm_tee.
    PhasmFrameRecording, PhasmTileRecording,
    // Phase B.1.1.a: per-AC-sign spatial metadata.
    AcSignMeta,
  };
  pub use crate::encoder::{
    encode_tile, FrameInvariants, FrameState,
    // W3.10.4: frame-level encode that returns OBU-wrapped bytes
    // + per-tile recorder data + tile_group offset from one call.
    // Enables phasm-core's av1_stego_encode flow without needing
    // the Context API plumbing (which would require generalizing
    // encode_normal_packet over the writer backend).
    encode_frame_with_phasm_tee,
    // D.6: helper used by encode_gop_with_phasm_tee to chain
    // reference frames between inter-encoded frames inside a GOP.
    update_rec_buffer,
  };
  pub use crate::stats::EncoderStats;

  // Helpers wrapping pub(crate) constructors so external callers
  // (phasm-core) don't need crate-internal access. Mirror what the
  // smoke test path inside src/encoder.rs::phasm_smoke_tests does.

  /// Construct an `InterConfig` from an `EncoderConfig`. Wraps the
  /// crate-private `InterConfig::new(&EncoderConfig)`.
  pub fn make_inter_config(
    enc_config: &crate::api::EncoderConfig,
  ) -> InterConfig {
    crate::api::PhasmInterConfig::new(enc_config)
  }

  /// Construct a default-padded `Frame<T>` for the given dimensions /
  /// chroma sampling. Wraps the crate-private `FrameAlloc::new` so
  /// callers don't need to compute LUMA_PADDING manually.
  pub fn make_frame<T: crate::util::Pixel>(
    width: usize,
    height: usize,
    chroma_sampling: crate::color::ChromaSampling,
  ) -> crate::frame::Frame<T> {
    <crate::frame::Frame<T> as crate::frame::FrameAlloc>::new(
      width,
      height,
      chroma_sampling,
    )
  }

  /// Read the `PHASM_AV1_LOOKAHEAD` env knob. Returns true when set to
  /// any non-zero parseable value. Default: false.
  ///
  /// **2026-06-29 (Option B Phase P2):** when enabled,
  /// [`encode_gop_with_phasm_tee`] runs a 1-frame self-refinement pass
  /// via [`crate::phasm_stego::compute_distortion_scales_for_window`]
  /// before each frame's encode. This pre-populates
  /// `lookahead_intra_costs` and writes refined `distortion_scales`
  /// into `fi.coded_frame_data` based on the per-block intra-cost
  /// estimate. The encoder picks this up during mode decision —
  /// content-dependent λ weighting rather than uniform default.
  ///
  /// Multi-frame propagation (stage 3 of the lookahead pipeline)
  /// requires constructing lookahead-frame FIs with proxy
  /// `rec_buffer`s that point at source frames (the trick rav1e CLI
  /// uses). That extension is deferred to a follow-on commit. 1-frame
  /// self-refinement already exercises stages 1, 2, and 4.
  ///
  /// When disabled (default), behaviour is byte-identical to the
  /// pre-P2 baseline — the existing AV1 byte-identity gates continue
  /// to pass without re-baselining.
  fn lookahead_enabled() -> bool {
    std::env::var("PHASM_AV1_LOOKAHEAD")
      .ok()
      .and_then(|s| s.parse::<u8>().ok())
      .map(|n| n != 0)
      .unwrap_or(false)
  }

  /// Run a 1-frame lookahead refinement pass on `(fi, fs, yuv)`
  /// before its `encode_frame_with_phasm_tee` call. Move-into-window
  /// + pop-back-out so the function's owning slice API
  /// (`compute_distortion_scales_for_window`) can mutate the FI/FS
  /// state in place.
  ///
  /// Stages 1 + 2 + 4 of the pipeline run on the 1-frame window;
  /// stage 3 (backwards propagation) is a no-op for n=1. The end
  /// effect is that `fi.coded_frame_data.distortion_scales` is
  /// rewritten from its uniform default to a per-block value derived
  /// from `lookahead_intra_costs` via `rdo::distortion_scale_for`.
  fn run_one_frame_lookahead<T: crate::util::Pixel>(
    fi: &mut FrameInvariants<T>,
    fs: &mut FrameState<T>,
    yuv: std::sync::Arc<crate::frame::Frame<T>>,
    output_frameno: u64,
    inter_cfg: &InterConfig,
    bit_depth: usize,
  ) {
    // Take owned fi + fs out of the caller's storage so we can move
    // them into the lookahead window. They get put back at the end.
    // The placeholders are constructed via uninit + immediate
    // overwrite — Rust enforces correct usage at type level.
    let owned_fi = std::mem::replace(
      fi,
      FrameInvariants::<T>::new_key_frame(
        fi.config.clone(),
        fi.sequence.clone(),
        0,
        Box::new([]),
      ),
    );
    let owned_fs = std::mem::replace(
      fs,
      FrameState::new_with_frame(fi, std::sync::Arc::clone(&yuv)),
    );

    let mut window =
      vec![crate::phasm_stego_lookahead::LookaheadWindowFrame {
        fi: owned_fi,
        fs: owned_fs,
        yuv,
        output_frameno,
      }];

    crate::phasm_stego_lookahead::compute_distortion_scales_for_window(
      &mut window,
      inter_cfg,
      bit_depth,
    );

    let crate::phasm_stego_lookahead::LookaheadWindowFrame {
      fi: refined_fi,
      fs: refined_fs,
      ..
    } = window.into_iter().next().expect("window had one entry");

    *fi = refined_fi;
    *fs = refined_fs;
  }

  /// D.6 — multi-frame GOP encode with WriterTee recording for stego.
  ///
  /// Encodes `yuvs` as a single GOP: frame 0 is the keyframe, frames
  /// 1..N are inter (P) frames referencing prior reconstructions. Each
  /// frame is encoded via [`encode_frame_with_phasm_tee`], so each
  /// returned `(packet, recording)` is wire-natural AV1 OBU bytes plus
  /// the phasm-core recorder data needed for STC override.
  ///
  /// The helper internalizes the reference-buffer chaining that the
  /// rav1e `Context::encode_normal_packet` does between frames (see
  /// `api/internal.rs:1448-1471`): pad reconstruction, call
  /// `update_rec_buffer`, copy `fi.rec_buffer` into the next frame's
  /// invariants, refresh `set_ref_frame_sign_bias`. Without this
  /// chain, inter frames have no references and either fail or fall
  /// back to intra-only (defeating the bitrate point of D.6).
  ///
  /// Low-latency mode is enforced inside the helper: B-frame
  /// reordering would require `frame_q`-based delayed emission which
  /// the per-GOP stego flow can't accommodate. Output order = input
  /// order, no SEFs, no reorder.
  ///
  /// **P2 (2026-06-29):** when `PHASM_AV1_LOOKAHEAD=1` is set, each
  /// frame's encode is preceded by a 1-frame lookahead refinement
  /// pass via [`run_one_frame_lookahead`]. Default (env unset or 0)
  /// is byte-identical to the pre-P2 baseline.
  pub fn encode_gop_with_phasm_tee<T: crate::util::Pixel>(
    yuvs: &[std::sync::Arc<crate::frame::Frame<T>>],
    config: std::sync::Arc<crate::api::EncoderConfig>,
    sequence: std::sync::Arc<crate::encoder::Sequence>,
  ) -> Vec<(Vec<u8>, PhasmFrameRecording<T>)> {
    assert!(!yuvs.is_empty(), "encode_gop_with_phasm_tee: empty GOP");
    let inter_cfg = make_inter_config(&config);
    let lookahead = lookahead_enabled();
    let bit_depth = config.bit_depth;

    let mut results = Vec::with_capacity(yuvs.len());

    // Frame 0 — keyframe.
    let mut prev_fi = FrameInvariants::<T>::new_key_frame(
      config.clone(),
      sequence.clone(),
      0,
      Box::new([]),
    );
    prev_fi.enable_segmentation = false;
    let mut fs = FrameState::new_with_frame(&prev_fi, yuvs[0].clone());
    if lookahead {
      run_one_frame_lookahead(
        &mut prev_fi,
        &mut fs,
        std::sync::Arc::clone(&yuvs[0]),
        0,
        &inter_cfg,
        bit_depth,
      );
    }
    let (packet, recording) =
      encode_frame_with_phasm_tee(&prev_fi, &mut fs, &inter_cfg);
    pad_and_update_ref(&mut prev_fi, &mut fs, 0);
    results.push((packet, recording));

    // Frames 1..N — inter (P) frames.
    let next_keyframe_input_frameno = yuvs.len() as u64;
    for (idx, yuv) in yuvs.iter().enumerate().skip(1) {
      let output_frameno_in_gop = idx as u64;

      let mut fi = FrameInvariants::<T>::new_inter_frame(
        &prev_fi,
        &inter_cfg,
        0,
        output_frameno_in_gop,
        next_keyframe_input_frameno,
        false,
        Box::new([]),
      )
      .expect("encode_gop_with_phasm_tee: new_inter_frame returned None");
      fi.enable_segmentation = false;
      fi.rec_buffer = prev_fi.rec_buffer.clone();
      fi.set_ref_frame_sign_bias();

      let mut fs = FrameState::new_with_frame(&fi, yuv.clone());
      if lookahead {
        run_one_frame_lookahead(
          &mut fi,
          &mut fs,
          std::sync::Arc::clone(yuv),
          output_frameno_in_gop,
          &inter_cfg,
          bit_depth,
        );
      }
      let (packet, recording) =
        encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);
      pad_and_update_ref(&mut fi, &mut fs, output_frameno_in_gop);
      results.push((packet, recording));
      prev_fi = fi;
    }

    results
  }

  fn pad_and_update_ref<T: crate::util::Pixel>(
    fi: &mut FrameInvariants<T>,
    fs: &mut FrameState<T>,
    output_frameno: u64,
  ) {
    let planes = if fi.sequence.chroma_sampling
      == crate::color::ChromaSampling::Cs400
    {
      1
    } else {
      3
    };
    use crate::frame::FramePad as _;
    // `encode_frame_with_phasm_tee` clones `fs.rec` into the returned
    // `PhasmFrameRecording.reconstructed_planes` (refcount becomes 2),
    // so `Arc::get_mut` would fail. `make_mut` copy-on-writes: it
    // allocates a new `Frame` with the same contents, swaps `fs.rec`
    // to point at it, and returns a unique mutable reference we can
    // pad in place. The recording keeps the original (unpadded)
    // Frame, which is correct — its consumer (J-UNIWARD cost
    // computation) reads the VISIBLE region only, and filter-tap
    // padding is outside that. The padded copy is what subsequent
    // frame's ME reads via `fi.rec_buffer`.
    //
    // Cost: one Frame-pixel copy per inter-frame chain step in a
    // GOP. ~3 MB per 1080p Y plane × gop_size frames. Bounded by
    // per-GOP memory budget (per phase-c-streaming-session-v6.md §10
    // ~100 MB ceiling at 1080p × 30f).
    std::sync::Arc::make_mut(&mut fs.rec).pad(fi.width, fi.height, planes);
    update_rec_buffer(output_frameno, fi, fs);
  }
}

pub(crate) mod built_info {
  // The file has been placed there by the build script.
  include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

mod serialize {
  cfg_if::cfg_if! {
    if #[cfg(feature="serialize")] {
      pub use serde::*;
    } else {
      pub use noop_proc_macro::{Deserialize, Serialize};
    }
  }
}

mod wasm_bindgen {
  cfg_if::cfg_if! {
    if #[cfg(feature="wasm")] {
      pub use wasm_bindgen::prelude::*;
    } else {
      pub use noop_proc_macro::wasm_bindgen;
    }
  }
}

#[cfg(any(cargo_c, feature = "capi"))]
pub mod capi;

#[macro_use]
mod transform;
#[macro_use]
mod cpu_features;

mod activity;
pub(crate) mod asm;
mod dist;
// phasm-stego patch: `pub` so downstream stego consumers (phasm-core)
// can implement `Writer` + `StorageBackend` and reuse `WriterRecorder`
// for Pass 1 record / Pass 2 cached-replay per
// docs/design/video/av1/streaming-session.md.
//
// Upstream is `mod ec;` (private). The trait + struct visibility
// inside the module (Writer, StorageBackend, WriterRecorder,
// WriterEncoder, WriterCounter, WriterCheckpoint) is already `pub`;
// only the module declaration needs widening.
pub mod ec;
mod partition;
mod predict;
mod quantize;
mod rdo;
mod rdo_tables;
#[macro_use]
mod util;
mod cdef;
#[doc(hidden)]
pub mod context;
mod deblock;
mod encoder;
mod entropymode;
mod levels;
mod lrf;
mod mc;
mod me;
mod rate;
mod recon_intra;
mod scan_order;
mod segmentation;
mod stats;
#[doc(hidden)]
pub mod tiling;
mod token_cdfs;

mod api;
mod frame;
mod header;

/// Commonly used types and traits.
pub mod prelude {
  pub use crate::api::*;
  pub use crate::encoder::{Sequence, Tune};
  pub use crate::frame::{
    Frame, FrameParameters, FrameTypeOverride, Plane, PlaneConfig,
  };
  pub use crate::partition::BlockSize;
  pub use crate::predict::PredictionMode;
  pub use crate::transform::TxType;
  pub use crate::util::{CastFromPrimitive, Pixel, PixelType};
}

/// Basic data structures
pub mod data {
  pub use crate::api::{
    ChromaticityPoint, EncoderStatus, FrameType, Packet, Rational,
  };
  pub use crate::frame::{Frame, FrameParameters};
  pub use crate::stats::EncoderStats;
  pub use crate::util::{CastFromPrimitive, Pixel, PixelType};
}

/// Encoder configuration and settings
pub mod config {
  pub use crate::api::config::{
    GrainTableSegment, NoiseGenArgs, TransferFunction, NUM_UV_COEFFS,
    NUM_UV_POINTS, NUM_Y_COEFFS, NUM_Y_POINTS,
  };
  pub use crate::api::{
    Config, EncoderConfig, InvalidConfig, PredictionModesSetting,
    RateControlConfig, RateControlError, RateControlSummary, SpeedSettings,
  };
  pub use crate::cpu_features::CpuFeatureLevel;
}

/// Version information
///
/// The information is recovered from `Cargo.toml` and `git describe`, when available.
///
/// ```
/// use rav1e::version;
/// use semver::Version;
///
/// let major = version::major();
/// let minor = version::minor();
/// let patch = version::patch();
///
/// let short = version::short();
///
/// let v1 = Version::new(major, minor, patch);
/// let v2 = Version::parse(&short).unwrap();
///
/// assert_eq!(v1.major, v2.major);
/// ```
pub mod version {
  /// Major version component
  ///
  /// It is increased every time a release presents a incompatible API change.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn major() -> u64 {
    env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap()
  }
  /// Minor version component
  ///
  /// It is increased every time a release presents new functionalities are added
  /// in a backwards-compatible manner.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn minor() -> u64 {
    env!("CARGO_PKG_VERSION_MINOR").parse().unwrap()
  }
  /// Patch version component
  ///
  /// It is increased every time a release provides only backwards-compatible bugfixes.
  ///
  /// # Panics
  ///
  /// Will panic if package is not built with Cargo,
  /// or if the package version is not a valid triplet of integers.
  pub fn patch() -> u64 {
    env!("CARGO_PKG_VERSION_PATCH").parse().unwrap()
  }

  /// Version information as presented in `[package]` `version`.
  ///
  /// e.g. `0.1.0`
  ///
  /// Can be parsed by [semver](https://crates.io/crates/semver).
  pub fn short() -> String {
    env!("CARGO_PKG_VERSION").to_string()
  }

  /// Version information as presented in `[package] version` followed by the
  /// short commit hash if present.
  ///
  /// e.g. `0.1.0 - g743d464`
  ///
  pub fn long() -> String {
    let s = short();
    let hash = hash();

    if hash.is_empty() {
      s
    } else {
      format!("{s} - {hash}")
    }
  }

  cfg_if::cfg_if! {
    if #[cfg(feature="git_version")] {
      fn git_version() -> &'static str {
        crate::built_info::GIT_VERSION.unwrap_or_default()
      }

      fn git_hash() -> &'static str {
        crate::built_info::GIT_COMMIT_HASH.unwrap_or_default()
      }
    } else {
      fn git_version() -> &'static str {
        "UNKNOWN"
      }

      fn git_hash() -> &'static str {
        "UNKNOWN"
      }
    }
  }
  /// Commit hash (short)
  ///
  /// Short hash of the git commit used by this build
  ///
  /// e.g. `g743d464`
  ///
  pub fn hash() -> String {
    git_hash().to_string()
  }

  /// Version information with the information
  /// provided by `git describe --tags`.
  ///
  /// e.g. `0.1.0 (v0.1.0-1-g743d464)`
  ///
  pub fn full() -> String {
    format!("{} ({})", short(), git_version(),)
  }
}
#[cfg(all(
  any(test, fuzzing),
  any(feature = "decode_test", feature = "decode_test_dav1d")
))]
mod test_encode_decode;

#[cfg(feature = "bench")]
pub mod bench {
  pub mod api {
    pub use crate::api::*;
  }
  pub mod cdef {
    pub use crate::cdef::*;
  }
  pub mod context {
    pub use crate::context::*;
  }
  pub mod dist {
    pub use crate::dist::*;
  }
  pub mod ec {
    pub use crate::ec::*;
  }
  pub mod encoder {
    pub use crate::encoder::*;
  }
  pub mod mc {
    pub use crate::mc::*;
  }
  pub mod partition {
    pub use crate::partition::*;
  }
  pub mod frame {
    pub use crate::frame::*;
  }
  pub mod predict {
    pub use crate::predict::*;
  }
  pub mod rdo {
    pub use crate::rdo::*;
  }
  pub mod tiling {
    pub use crate::tiling::*;
  }
  pub mod transform {
    pub use crate::transform::*;
  }
  pub mod util {
    pub use crate::util::*;
  }
  pub mod cpu_features {
    pub use crate::cpu_features::*;
  }
}

#[cfg(fuzzing)]
pub mod fuzzing;
