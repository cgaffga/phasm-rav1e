// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: BSD-2-Clause
//
// phasm-stego streaming-session lookahead (Option B Phase P1)
//
// Free-function counterpart to `ContextInner::compute_block_importances`
// (`api/internal.rs:1075-1259`). The standard rav1e `Context::receive_packet`
// pull-API runs a temporal-RDO lookahead pass internally: for every output
// frame it walks its `frame_data` BTreeMap backwards from the latest
// lookahead frame and propagates per-block importance into each reference
// frame's accumulator, then converts the current frame's accumulator into
// the per-block `distortion_scales` array that the encoder reads during
// mode decision.
//
// phasm's streaming-session encode path (`encode_gop_with_phasm_tee`)
// bypasses the entire `Context` machinery — it needs a writer tee, per-GOP
// byte-deterministic replay, single-GOP-RAM streaming-pull semantics, and
// output-order==input-order ([speed-preset audit § 6][sp]). That bypass
// also skips the lookahead refinement, which the audit identified as the
// dominant structural source of the phasm-vs-rav1e-CLI Layer-3 partition
// fingerprint divergence (BS effect-size 0.110 at s9/Q100 carplane).
//
// This module re-implements the lookahead refinement as a free function
// over a flat slice of `LookaheadWindowFrame`, so phasm-core's
// streaming-session encode can compute refined `distortion_scales` for
// each frame without inheriting `Context`'s push/pull state machine.
//
// Design source: extracted from `internal.rs:1075-1259` (commit at extraction
// time — see git blame). Future rav1e upstream merges should check this file
// for drift against the original; if `compute_block_importances` or
// `update_block_importances` evolve upstream, the corresponding logic here
// needs to follow.
//
// [sp]: ../../docs/design/video/av1/stealth-audit-speed-preset-2026-06-29.md
//
// Implementation phases (#232 = this file; tracked in
// av1-stealth-lookahead-plan-2026-06-29.md § 3.3):
//
//   P1.a (this commit): module skeleton + signature + smoke test scaffold.
//   P1.b: stage 1 — compute_motion_vectors per frame.
//   P1.c: stage 2 — lookahead_intra_costs per non-keyframe.
//   P1.d: stage 3 — backwards propagation loop.
//   P1.e: stage 4 — distortion_scales finalization for window[0].
//   P1.f: validation — smoke test asserts non-trivial distortion_scales,
//         and a byte-comparison gate against a rav1e-CLI reference run
//         with the same lookahead window.

use std::sync::Arc;

use crate::api::PhasmInterConfig as InterConfig;
use crate::api::FrameType;
use crate::encoder::{FrameInvariants, FrameState};
use crate::frame::Frame;
use crate::util::Pixel;

/// One frame in a lookahead window. The window is a contiguous slice in
/// chronological order: `window[0]` is the frame that will be encoded
/// next; `window[1..]` are future frames used as lookahead context.
///
/// On input, the caller must have constructed `fi` + `fs` for each frame
/// the same way `encode_gop_with_phasm_tee` already does
/// (`FrameInvariants::new_key_frame` / `new_inter_frame` + a fresh
/// `FrameState::new_with_frame`). The `yuv` Arc is shared with `fs`
/// (FrameState already holds a Frame Arc internally; we keep a separate
/// reference so the propagation loop can access the raw YUV without
/// peeking inside FrameState's accessor surface).
///
/// On output, `compute_distortion_scales_for_window` will populate (or
/// overwrite) the `distortion_scales` field of `window[0].fi
/// .coded_frame_data` so that a subsequent `encode_frame_with_phasm_tee`
/// call uses the lookahead-refined RDO weights. `block_importances` on
/// every frame is treated as scratch and overwritten.
pub struct LookaheadWindowFrame<T: Pixel> {
    pub fi: FrameInvariants<T>,
    pub fs: FrameState<T>,
    pub yuv: Arc<Frame<T>>,
    /// Display-order index of this frame in the encode. Used as the key
    /// when the propagation loop needs to look up "the frame that
    /// `window[k].fi.ref_frames[r]` refers to". Caller assigns it
    /// (typically the input frameno in the GOP).
    pub output_frameno: u64,
}

/// Compute lookahead-refined per-block `distortion_scales` for
/// `window[0]` using `window[1..]` as the lookahead context.
///
/// Mirror of `ContextInner::compute_block_importances` from
/// `api/internal.rs:1075-1259`, adapted to operate on a flat slice.
///
/// # Stages
///
/// 1. **Motion estimation.** For every non-keyframe in the window, run
///    `compute_motion_vectors` to populate `fs.frame_me_stats`. This is
///    the same per-frame ME pass `ContextInner::compute_frame_invariants`
///    runs during lookahead intake.
///
/// 2. **Intra-cost estimation.** For every non-keyframe, populate
///    `fi.coded_frame_data.lookahead_intra_costs` via
///    `estimate_intra_costs`. (Keyframes don't propagate importance
///    into earlier frames, so they don't need this.)
///
/// 3. **Backwards propagation.** Walk the window from latest to second
///    (skipping `window[0]`), and for each non-keyframe call
///    `update_block_importances` once per unique reference. Each call
///    propagates importance from the current frame's
///    `block_importances` accumulator (initialized to zero) into the
///    referenced frame's accumulator, weighted by the precomputed ME
///    stats and the intra-vs-inter cost gap.
///
/// 4. **Finalization.** Convert `window[0].fi.coded_frame_data
///    .block_importances` (now accumulated from all later frames) into
///    `distortion_scales` via `crate::rdo::distortion_scale_for`. This
///    is the value the encoder reads during mode decision.
///
/// # Skeleton status (2026-06-29)
///
/// This is the P1.a checkpoint — the function signature + module wiring
/// is in place but the body is a no-op (the four stages are TODO).
/// Calling this function today is safe but produces no refinement; the
/// caller will get the same encode it would have gotten without
/// lookahead. P1.b-e fill in the stages incrementally.
pub fn compute_distortion_scales_for_window<T: Pixel>(
    window: &mut [LookaheadWindowFrame<T>],
    inter_cfg: &InterConfig,
    _bit_depth: usize,
) {
    if window.is_empty() {
        return;
    }

    // Stage 1 (P1.b 2026-06-29): per-frame motion estimation.
    //
    // Populates `fs.frame_me_stats` for each non-keyframe in the
    // window. Mirror of what `ContextInner::compute_frame_invariants`
    // (`api/internal.rs:898-908`) does during lookahead intake: it
    // calls `compute_lookahead_motion_vectors` once per frame as the
    // frame enters the lookahead window. We do the same here, except
    // straight on the window slice instead of through the BTreeMap.
    //
    // Keyframes are skipped — they have no inter prediction, so no
    // me_stats are needed and `compute_motion_vectors` would do
    // wasted work on them anyway.
    //
    // `fi.rec_buffer` MUST be populated before this call for inter
    // frames (the ME pass uses `rec_buffer.frames[r]` to fetch each
    // reference). Caller's responsibility — `encode_gop_with_phasm_tee`
    // already chains `rec_buffer` between frames via
    // `pad_and_update_ref`.
    for frame in window.iter_mut() {
        if frame.fi.frame_type != FrameType::KEY {
            crate::api::lookahead::compute_motion_vectors(
                &mut frame.fi,
                &mut frame.fs,
                inter_cfg,
            );
        }
    }

    // TODO P1.c: stage 2 — for each non-keyframe in window:
    // TODO P1.c: stage 2 — for each non-keyframe in window, populate
    //     window[k].fi.coded_frame_data.lookahead_intra_costs via
    //     crate::api::lookahead::estimate_intra_costs(...). Mirror of
    //     ContextInner::compute_lookahead_intra_costs at
    //     internal.rs:838-877.
    //
    // TODO P1.d: stage 3 — backwards propagation loop (mirror of
    //     internal.rs:1095-1209). For each output_frameno from latest
    //     down to window[1].output_frameno: for each unique reference,
    //     call update_block_importances(...) once. The reference frame
    //     lookup goes through window.iter_mut().find(|f| f.output_frameno
    //     == ref_n) instead of self.frame_data.get_mut(&ref_n).
    //
    // TODO P1.e: stage 4 — finalization (mirror of internal.rs:1211-1230).
    //     For window[0]: zip block_importances + lookahead_intra_costs +
    //     distortion_scales, write distortion_scale_for(propagate_cost,
    //     intra_cost) into each slot.
}

/// Internal helper extracted from `ContextInner::update_block_importances`
/// (`api/internal.rs:912-1071`). Same signature except no `&self` —
/// the original was a method that didn't read `self` (all state came
/// through arguments), so it's a clean lift.
///
/// P1.a checkpoint: stub only. Body lands in P1.d alongside the
/// propagation loop.
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn update_block_importances_free<T: Pixel>(
    _fi: &FrameInvariants<T>,
    _me_stats: &crate::me::FrameMEStats,
    _frame: &Frame<T>,
    _reference_frame: &Frame<T>,
    _bit_depth: usize,
    _bsize: crate::partition::BlockSize,
    _len: usize,
    _reference_frame_block_importances: &mut [f32],
) {
    // TODO P1.d: copy the body of ContextInner::update_block_importances
    // verbatim (api/internal.rs:912-1071). The method body uses no
    // `&self`, so the lift is mechanical.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P1.a smoke test: function is callable from the phasm_stego
    /// re-export and an empty window doesn't panic. Real behavioural
    /// gates land in P1.f.
    #[test]
    fn empty_window_is_noop() {
        let mut window: Vec<LookaheadWindowFrame<u8>> = Vec::new();
        // We need an InterConfig to call; construct a default one via
        // the smallest-possible EncoderConfig.
        let mut cfg = crate::api::EncoderConfig::default();
        cfg.width = 16;
        cfg.height = 16;
        let inter_cfg = crate::api::PhasmInterConfig::new(&cfg);
        compute_distortion_scales_for_window(&mut window, &inter_cfg, 8);
        assert!(window.is_empty());
    }
}
