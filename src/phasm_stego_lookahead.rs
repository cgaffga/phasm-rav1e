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

use std::mem;
use std::sync::Arc;

use arrayvec::ArrayVec;

use crate::api::PhasmInterConfig as InterConfig;
use crate::api::FrameType;
use crate::api::lookahead::{
    IMP_BLOCK_AREA_IN_MV_UNITS, IMP_BLOCK_MV_UNITS_PER_PIXEL,
    IMP_BLOCK_SIZE_IN_MV_UNITS,
};
use crate::dist::get_satd;
use crate::encoder::{FrameInvariants, FrameState, IMPORTANCE_BLOCK_SIZE};
use crate::frame::{AsRegion, Frame};
use crate::partition::BlockSize;
use crate::tiling::Area;
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
    // Stage 2 (P1.c 2026-06-29, refined post-P1.e): per-frame
    // intra-cost estimation.
    //
    // For each non-SEF frame in the window — **including keyframes** —
    // populate `fi.coded_frame_data.lookahead_intra_costs` via
    // `estimate_intra_costs`. Mirror of
    // `ContextInner::compute_lookahead_intra_costs`
    // (`api/internal.rs:838-877`), which gates only on
    // `is_show_existing_frame`, NOT on `frame_type`.
    //
    // Keyframes DO need this populated because stage 4 (finalization)
    // reads window[0]'s lookahead_intra_costs unconditionally — and in
    // phasm's streaming-session model window[0] is the IDR for the
    // first frame of every GOP. Skipping keyframes here was a P1.c
    // first-draft bug fixed at the P1.e checkpoint.
    //
    // Differences from rav1e CLI path:
    //   - rav1e's ContextInner caches per-frame intra-cost arrays in
    //     `keyframe_detector.intra_costs` during scene-change detection
    //     and pulls them out lazily. Phasm has no scene-change detector
    //     in the streaming session (low_latency=true, no scenecut
    //     analysis), so we always go down the fresh-compute branch
    //     (the `unwrap_or_else` arm in the original). Slightly more
    //     work per frame than rav1e CLI but functionally identical.
    //
    //   - `is_show_existing_frame` check is preserved out of paranoia:
    //     phasm's streaming session encodes one tile group per real
    //     frame and never emits SEFs, so this branch is dead code
    //     today. Keeping it makes the function defensive against a
    //     future refactor.
    for frame in window.iter_mut() {
        if frame.fi.is_show_existing_frame() {
            continue;
        }
        let bit_depth = frame.fi.sequence.bit_depth;
        let cpu_feature_level = frame.fi.cpu_feature_level;
        let mut temp_plane = frame.yuv.planes[0].clone();
        let intra_costs = crate::api::lookahead::estimate_intra_costs(
            &mut temp_plane,
            &frame.yuv,
            bit_depth,
            cpu_feature_level,
        );
        frame
            .fi
            .coded_frame_data
            .as_mut()
            .expect("coded_frame_data must be set after new_key_frame / new_inter_frame")
            .lookahead_intra_costs = intra_costs;
    }

    // Stage 3 (P1.d.2 2026-06-29): backwards importance propagation.
    //
    // Mirror of internal.rs:1095-1209. Walk the window from latest to
    // window[1] (skip window[0] — its block_importances accumulate
    // from every later frame; the finalization in stage 4 reads it).
    // For each non-keyframe k, find the unique references and for each
    // call update_block_importances_free, which propagates the
    // importance back into the referenced frame's accumulator.
    //
    // Borrow-checker note: the original uses BTreeMap::remove + insert
    // to break the aliasing between the current frame's `fi` and the
    // referenced frame's `block_importances` mutation. Slices can't
    // remove. We use std::mem::take instead — swap the referenced
    // frame's `block_importances` Box<[f32]> with the Default
    // (an empty boxed slice), do the propagation against the taken
    // buffer (now an owned local), then put it back. Zero-allocation
    // at the Box level — just moves the buffer ptr.
    let bsize = BlockSize::from_width_and_height(
        IMPORTANCE_BLOCK_SIZE,
        IMPORTANCE_BLOCK_SIZE,
    );

    // Initialize block_importances to 0 on every frame in window.
    for frame in window.iter_mut() {
        if let Some(coded) = frame.fi.coded_frame_data.as_mut() {
            for x in coded.block_importances.iter_mut() {
                *x = 0.0;
            }
        }
    }

    let output_framenos: Vec<u64> =
        window.iter().map(|f| f.output_frameno).collect();
    let n = output_framenos.len();

    for k in (1..n).rev() {
        if window[k].fi.frame_type == FrameType::KEY {
            continue;
        }

        // Collect unique reference indices (≤ 3 per AV1 spec).
        let mut unique_indices: ArrayVec<(usize, u8), 3> = ArrayVec::new();
        for (mv_index, &rec_index) in window[k].fi.ref_frames.iter().enumerate()
        {
            if !unique_indices.iter().any(|&(_, r)| r == rec_index) {
                unique_indices.push((mv_index, rec_index));
            }
        }
        let bit_depth = window[k].fi.sequence.bit_depth;
        let len = unique_indices.len();

        // Snapshot the me_stats Arc so the guard's lifetime is
        // independent of window borrows below.
        let me_stats_arc = window[k].fs.frame_me_stats.clone();
        let me_stats_guard = me_stats_arc.read().expect("poisoned lock");

        for &(mv_index, rec_index) in unique_indices.iter() {
            let reference_arc =
                match window[k].fi.rec_buffer.frames[rec_index as usize].as_ref()
                {
                    Some(r) => Arc::clone(r),
                    None => continue,
                };
            let ref_output_frameno = reference_arc.output_frameno;
            let reference_frame_arc = Arc::clone(&reference_arc.frame);

            debug_assert_ne!(ref_output_frameno, window[k].output_frameno);

            let ref_idx = match output_framenos
                .iter()
                .position(|&m| m == ref_output_frameno)
            {
                Some(i) => i,
                None => continue,
            };
            if ref_idx == k {
                continue;
            }

            // Swap ref's block_importances out of the window so we can
            // immutably borrow window[k] without aliasing window[ref_idx].
            let mut taken_bi =
                match window[ref_idx].fi.coded_frame_data.as_mut() {
                    Some(c) => mem::take(&mut c.block_importances),
                    None => continue,
                };

            update_block_importances_free(
                &window[k].fi,
                &me_stats_guard[mv_index],
                &window[k].yuv,
                &reference_frame_arc,
                bit_depth,
                bsize,
                len,
                &mut taken_bi,
            );

            // Put it back.
            window[ref_idx]
                .fi
                .coded_frame_data
                .as_mut()
                .expect("coded_frame_data was set when we took block_importances")
                .block_importances = taken_bi;
        }

        drop(me_stats_guard);
    }

    // Stage 4 (P1.e 2026-06-29): finalization for window[0].
    //
    // Convert window[0]'s accumulated block_importances + lookahead
    // intra-costs into the per-block `distortion_scales` the encoder
    // reads during mode decision. Mirror of internal.rs:1211-1230.
    if let Some(coded) = window[0].fi.coded_frame_data.as_mut() {
        let block_importances = coded.block_importances.iter();
        let lookahead_intra_costs = coded.lookahead_intra_costs.iter();
        let distortion_scales = coded.distortion_scales.iter_mut();
        for ((&propagate_cost, &intra_cost), distortion_scale) in
            block_importances.zip(lookahead_intra_costs).zip(distortion_scales)
        {
            *distortion_scale = crate::rdo::distortion_scale_for(
                propagate_cost as f64,
                intra_cost as f64,
            );
        }
    }
}

/// Internal helper extracted from `ContextInner::update_block_importances`
/// (`api/internal.rs:912-1071`). Same signature except no `&self` —
/// the original was a method that didn't read `self` (all state came
/// through arguments), so it's a clean lift.
///
/// Propagates importance from `fi`'s block_importances accumulator into
/// `reference_frame_block_importances` via the per-MB MV ME stats. Per
/// (impl-block, current-MV-block):
///
/// 1. Compute SATD of current MV-block against the referenced location
///    using `me_stats[mv_index]`.
/// 2. Compute `propagate_fraction = max(0, 1 - inter_cost / intra_cost)`
///    — how much of the importance "flows backward" through this ref.
/// 3. Split the propagated amount across the 4 reference-grid blocks
///    that the MV-block straddles (bilinear-ish — fractions weighted by
///    overlap area in MV-unit space).
///
/// Body verbatim from internal.rs:912-1071 with `Self::` → free and
/// no other changes. Keep this in sync with the original when rav1e
/// upstream evolves the function.
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_block_importances_free<T: Pixel>(
    fi: &FrameInvariants<T>,
    me_stats: &crate::me::FrameMEStats,
    frame: &Frame<T>,
    reference_frame: &Frame<T>,
    bit_depth: usize,
    bsize: BlockSize,
    len: usize,
    reference_frame_block_importances: &mut [f32],
) {
    let coded_data = fi.coded_frame_data.as_ref().unwrap();
    let plane_org = &frame.planes[0];
    let plane_ref = &reference_frame.planes[0];
    let lookahead_intra_costs_lines =
        coded_data.lookahead_intra_costs.chunks_exact(coded_data.w_in_imp_b);
    let block_importances_lines =
        coded_data.block_importances.chunks_exact(coded_data.w_in_imp_b);

    lookahead_intra_costs_lines
        .zip(block_importances_lines)
        .zip(me_stats.rows_iter().step_by(2))
        .enumerate()
        .flat_map(
            |(y, ((lookahead_intra_costs, block_importances), me_stats_line))| {
                lookahead_intra_costs
                    .iter()
                    .zip(block_importances.iter())
                    .zip(me_stats_line.iter().step_by(2))
                    .enumerate()
                    .map(move |(x, ((&intra_cost, &future_importance), &me_stat))| {
                        let mv = me_stat.mv;

                        // Coordinates of the top-left corner of the reference block,
                        // in MV units.
                        let reference_x =
                            x as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.col as i64;
                        let reference_y =
                            y as i64 * IMP_BLOCK_SIZE_IN_MV_UNITS + mv.row as i64;

                        let region_org = plane_org.region(Area::Rect {
                            x: (x * IMPORTANCE_BLOCK_SIZE) as isize,
                            y: (y * IMPORTANCE_BLOCK_SIZE) as isize,
                            width: IMPORTANCE_BLOCK_SIZE,
                            height: IMPORTANCE_BLOCK_SIZE,
                        });

                        let region_ref = plane_ref.region(Area::Rect {
                            x: reference_x as isize
                                / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                            y: reference_y as isize
                                / IMP_BLOCK_MV_UNITS_PER_PIXEL as isize,
                            width: IMPORTANCE_BLOCK_SIZE,
                            height: IMPORTANCE_BLOCK_SIZE,
                        });

                        let inter_cost = get_satd(
                            &region_org,
                            &region_ref,
                            bsize.width(),
                            bsize.height(),
                            bit_depth,
                            fi.cpu_feature_level,
                        ) as f32;

                        let intra_cost = intra_cost as f32;

                        let propagate_fraction = if intra_cost <= inter_cost {
                            0.
                        } else {
                            1. - inter_cost / intra_cost
                        };

                        let propagate_amount = (intra_cost + future_importance)
                            * propagate_fraction
                            / len as f32;
                        (propagate_amount, reference_x, reference_y)
                    })
            },
        )
        .for_each(|(propagate_amount, reference_x, reference_y)| {
            let mut propagate =
                |block_x_in_mv_units, block_y_in_mv_units, fraction| {
                    let x = block_x_in_mv_units / IMP_BLOCK_SIZE_IN_MV_UNITS;
                    let y = block_y_in_mv_units / IMP_BLOCK_SIZE_IN_MV_UNITS;

                    // TODO: propagate partially if the block is partially off-frame
                    // (possible on right and bottom edges)?
                    if x >= 0
                        && y >= 0
                        && (x as usize) < coded_data.w_in_imp_b
                        && (y as usize) < coded_data.h_in_imp_b
                    {
                        reference_frame_block_importances
                            [y as usize * coded_data.w_in_imp_b + x as usize] +=
                            propagate_amount * fraction;
                    }
                };

            // Coordinates of the top-left corner of the block intersecting the
            // reference block from the top-left.
            let top_left_block_x = (reference_x
                - if reference_x < 0 { IMP_BLOCK_SIZE_IN_MV_UNITS - 1 } else { 0 })
                / IMP_BLOCK_SIZE_IN_MV_UNITS
                * IMP_BLOCK_SIZE_IN_MV_UNITS;
            let top_left_block_y = (reference_y
                - if reference_y < 0 { IMP_BLOCK_SIZE_IN_MV_UNITS - 1 } else { 0 })
                / IMP_BLOCK_SIZE_IN_MV_UNITS
                * IMP_BLOCK_SIZE_IN_MV_UNITS;

            debug_assert!(reference_x >= top_left_block_x);
            debug_assert!(reference_y >= top_left_block_y);

            let top_right_block_x = top_left_block_x + IMP_BLOCK_SIZE_IN_MV_UNITS;
            let top_right_block_y = top_left_block_y;
            let bottom_left_block_x = top_left_block_x;
            let bottom_left_block_y =
                top_left_block_y + IMP_BLOCK_SIZE_IN_MV_UNITS;
            let bottom_right_block_x = top_right_block_x;
            let bottom_right_block_y = bottom_left_block_y;

            let top_left_block_fraction = ((top_right_block_x - reference_x)
                * (bottom_left_block_y - reference_y))
                as f32
                / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

            propagate(top_left_block_x, top_left_block_y, top_left_block_fraction);

            let top_right_block_fraction =
                ((reference_x + IMP_BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                    * (bottom_left_block_y - reference_y)) as f32
                    / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

            propagate(
                top_right_block_x,
                top_right_block_y,
                top_right_block_fraction,
            );

            let bottom_left_block_fraction = ((top_right_block_x - reference_x)
                * (reference_y + IMP_BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
                as f32
                / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

            propagate(
                bottom_left_block_x,
                bottom_left_block_y,
                bottom_left_block_fraction,
            );

            let bottom_right_block_fraction =
                ((reference_x + IMP_BLOCK_SIZE_IN_MV_UNITS - top_right_block_x)
                    * (reference_y + IMP_BLOCK_SIZE_IN_MV_UNITS - bottom_left_block_y))
                    as f32
                    / IMP_BLOCK_AREA_IN_MV_UNITS as f32;

            propagate(
                bottom_right_block_x,
                bottom_right_block_y,
                bottom_right_block_fraction,
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ChromaSampling;
    use crate::encoder::Sequence;
    use crate::frame::FrameAlloc;

    fn make_default_config(w: usize, h: usize) -> Arc<crate::api::EncoderConfig> {
        let mut cfg = crate::api::EncoderConfig::default();
        cfg.width = w;
        cfg.height = h;
        cfg.bit_depth = 8;
        cfg.chroma_sampling = ChromaSampling::Cs420;
        // Mirror what phasm-core's encode_gop_natural pins for the
        // streaming-session path (whole_video.rs::encode_gop_natural).
        // Without low_latency=true, new_inter_frame returns None
        // because the default config schedules B-frame reordering.
        cfg.low_latency = true;
        Arc::new(cfg)
    }

    /// P1.a smoke test: function is callable from the phasm_stego
    /// re-export and an empty window doesn't panic.
    #[test]
    fn empty_window_is_noop() {
        let mut window: Vec<LookaheadWindowFrame<u8>> = Vec::new();
        let cfg = make_default_config(16, 16);
        let inter_cfg = crate::api::PhasmInterConfig::new(&cfg);
        compute_distortion_scales_for_window(&mut window, &inter_cfg, 8);
        assert!(window.is_empty());
    }

    /// P1.f behavioural test: stage 2 (intra-cost estimation) must
    /// populate `lookahead_intra_costs` on **keyframes** too — not
    /// just inter frames.
    ///
    /// In phasm's streaming-session model, `window[0]` is the IDR for
    /// the first frame of every GOP. Stage 4 (finalization) reads
    /// `window[0].fi.coded_frame_data.lookahead_intra_costs`
    /// unconditionally — if stage 2 skipped the keyframe, the array
    /// would stay at its `Box::new([])` default and stage 4 would be
    /// a no-op (zipping an empty iterator with `distortion_scales`).
    ///
    /// rav1e's own `compute_lookahead_intra_costs`
    /// (`api/internal.rs:838-877`) only gates on
    /// `is_show_existing_frame`, NOT `frame_type == KEY` — so for
    /// behaviour-parity we must do the same.
    ///
    /// This test catches a P1.c first-draft bug where stage 2 had a
    /// `if frame_type == KEY { continue; }` filter (fixed at the
    /// P1.e checkpoint).
    #[test]
    fn keyframe_window_populates_lookahead_intra_costs() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = make_default_config(W, H);
        let sequence = Arc::new(Sequence::new(&cfg));
        let inter_cfg = crate::api::PhasmInterConfig::new(&cfg);

        let yuv: Arc<Frame<u8>> = Arc::new(
            <Frame<u8> as FrameAlloc>::new(W, H, ChromaSampling::Cs420),
        );

        let mut fi = FrameInvariants::<u8>::new_key_frame(
            cfg.clone(),
            sequence.clone(),
            0,
            Box::new([]),
        );
        fi.enable_segmentation = false;
        let fs = FrameState::new_with_frame(&fi, Arc::clone(&yuv));

        let mut window = vec![LookaheadWindowFrame {
            fi,
            fs,
            yuv,
            output_frameno: 0,
        }];

        // Before: CodedFrameData::new initializes lookahead_intra_costs
        // to Box::new([]).
        let initial_len = window[0]
            .fi
            .coded_frame_data
            .as_ref()
            .expect("KEY frame should have coded_frame_data")
            .lookahead_intra_costs
            .len();
        assert_eq!(
            initial_len, 0,
            "lookahead_intra_costs should start empty on a fresh KEY frame"
        );

        compute_distortion_scales_for_window(&mut window, &inter_cfg, 8);

        // After: stage 2 should have populated lookahead_intra_costs
        // to w_in_imp_b * h_in_imp_b entries on the KEY frame.
        let coded = window[0].fi.coded_frame_data.as_ref().unwrap();
        let after_len = coded.lookahead_intra_costs.len();
        let expected_len = coded.w_in_imp_b * coded.h_in_imp_b;
        assert_eq!(
            after_len, expected_len,
            "stage 2 should populate lookahead_intra_costs on KEY frames \
             (w_in_imp_b={} h_in_imp_b={} → expected={} entries, got={})",
            coded.w_in_imp_b, coded.h_in_imp_b, expected_len, after_len
        );

        // Also: block_importances must remain w_in_imp_b * h_in_imp_b
        // (initialized by CodedFrameData::new; stage 3's zero-init
        // pass should preserve length).
        assert_eq!(
            coded.block_importances.len(),
            expected_len,
            "block_importances length should match w_in_imp_b * h_in_imp_b"
        );
    }

    /// P2 smoke test: with `PHASM_AV1_LOOKAHEAD=1` set,
    /// `encode_gop_with_phasm_tee` runs the 1-frame self-refinement
    /// pass via `run_one_frame_lookahead` (the move-into-window +
    /// pop-back-out placeholder dance) before each frame's encode
    /// without panicking.
    ///
    /// This is a crash-safety test only — it doesn't assert anything
    /// about the encoded bytes (cargo test env-var sharing across
    /// parallel tests is unsafe, so we can't reliably bracket an
    /// env-off encode and an env-on encode in the same test).
    /// Behavioural validation of the lookahead refinement comes via
    /// P5's stealth audit re-run on a fresh CLI invocation with the
    /// env knob set.
    #[test]
    fn p2_env_knob_on_does_not_crash() {
        const W: usize = 64;
        const H: usize = 64;
        let cfg = make_default_config(W, H);
        let sequence = Arc::new(Sequence::new(&cfg));

        let yuvs: Vec<Arc<Frame<u8>>> = (0..2)
            .map(|_| {
                Arc::new(<Frame<u8> as FrameAlloc>::new(
                    W,
                    H,
                    ChromaSampling::Cs420,
                ))
            })
            .collect();

        // SAFETY: set_var is unsafe in 2024 edition; this test must
        // run with --test-threads=1 to avoid racing with any future
        // test that also touches this env var. Today no other test
        // reads PHASM_AV1_LOOKAHEAD.
        std::env::set_var("PHASM_AV1_LOOKAHEAD", "1");
        let results = crate::phasm_stego::encode_gop_with_phasm_tee::<u8>(
            &yuvs,
            cfg,
            sequence,
        );
        std::env::remove_var("PHASM_AV1_LOOKAHEAD");

        assert_eq!(
            results.len(),
            2,
            "encode_gop_with_phasm_tee should return one (packet, recording) pair per frame"
        );
        assert!(
            !results[0].0.is_empty(),
            "frame 0 packet bytes should be non-empty"
        );
        assert!(
            !results[1].0.is_empty(),
            "frame 1 packet bytes should be non-empty"
        );
    }
}
