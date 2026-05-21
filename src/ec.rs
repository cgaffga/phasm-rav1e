// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

#![allow(non_camel_case_types)]

cfg_if::cfg_if! {
  if #[cfg(nasm_x86_64)] {
    pub use crate::asm::x86::ec::*;
  } else {
    pub use self::rust::*;
  }
}

use crate::context::{CDFContext, CDFContextLog, CDFOffset};
use bitstream_io::{BigEndian, BitWrite, BitWriter};
use std::io;

pub const OD_BITRES: u8 = 3;
const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
type ec_window = u32;

/// Public trait interface to a bitstream `Writer`: a `Counter` can be
/// used to count bits for cost analysis without actually storing
/// anything (using a new `WriterCounter` as a `Writer`), to record
/// tokens for later writing (using a new `WriterRecorder` as a
/// `Writer`) to write actual final bits out using a range encoder
/// (using a new `WriterEncoder` as a `Writer`).  A `WriterRecorder`'s
/// contents can be replayed into a `WriterEncoder`.
pub trait Writer {
  /// Write a symbol `s`, using the passed in cdf reference; leaves `cdf` unchanged
  fn symbol<const CDF_LEN: usize>(&mut self, s: u32, cdf: &[u16; CDF_LEN]);
  /// return approximate number of fractional bits in `OD_BITRES`
  /// precision to write a symbol `s` using the passed in cdf reference;
  /// leaves `cdf` unchanged
  fn symbol_bits(&self, s: u32, cdf: &[u16]) -> u32;
  /// Write a symbol `s`, using the passed in cdf reference; updates the referenced cdf.
  fn symbol_with_update<const CDF_LEN: usize>(
    &mut self, s: u32, cdf: CDFOffset<CDF_LEN>, log: &mut CDFContextLog,
    fc: &mut CDFContext,
  );
  /// Write a bool using passed in probability
  fn bool(&mut self, val: bool, f: u16);
  /// Write a single bit with flat probability
  fn bit(&mut self, bit: u16);
  /// Write literal `bits` with flat probability
  fn literal(&mut self, bits: u8, s: u32);
  /// Write passed `level` as a golomb code
  fn write_golomb(&mut self, level: u32);
  /// Write a value `v` in `[0, n-1]` quasi-uniformly
  fn write_quniform(&mut self, n: u32, v: u32);
  /// Return fractional bits needed to write a value `v` in `[0, n-1]`
  /// quasi-uniformly
  fn count_quniform(&self, n: u32, v: u32) -> u32;
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite subexponential
  fn write_subexp(&mut self, n: u32, k: u8, v: u32);
  /// Return fractional bits needed to write symbol v in `[0, n-1]` with
  /// parameter k as finite subexponential
  fn count_subexp(&self, n: u32, k: u8, v: u32) -> u32;
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[0, n-1]`.
  fn write_unsigned_subexp_with_ref(&mut self, v: u32, mx: u32, k: u8, r: u32);
  /// Return fractional bits needed to write symbol `v` in `[0, n-1]` with
  /// parameter `k` as finite subexponential based on a reference `r`
  /// also in `[0, n-1]`.
  fn count_unsigned_subexp_with_ref(
    &self, v: u32, mx: u32, k: u8, r: u32,
  ) -> u32;
  /// Write symbol v in `[-(n-1), n-1]` with parameter k as finite
  /// subexponential based on a reference ref also in `[-(n-1), n-1]`.
  fn write_signed_subexp_with_ref(
    &mut self, v: i32, low: i32, high: i32, k: u8, r: i32,
  );
  /// Return fractional bits needed to write symbol `v` in `[-(n-1), n-1]`
  /// with parameter `k` as finite subexponential based on a reference
  /// `r` also in `[-(n-1), n-1]`.
  fn count_signed_subexp_with_ref(
    &self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) -> u32;
  /// Return current length of range-coded bitstream in integer bits
  fn tell(&mut self) -> u32;
  /// Return current length of range-coded bitstream in fractional
  /// bits with `OD_BITRES` decimal precision
  fn tell_frac(&mut self) -> u32;
  /// Save current point in coding/recording to a checkpoint
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Restore saved position in coding/recording from a checkpoint
  fn rollback(&mut self, _: &WriterCheckpoint);
  /// Add additional bits from rate estimators without coding a real symbol
  fn add_bits_frac(&mut self, bits_frac: u32);
  /// phasm-stego (fork, W3.9.0): set the per-channel emission tag for
  /// subsequent 50/50 binary emissions (`bit()` / `bool(_, 16384)` /
  /// `literal()` / `write_golomb()`). Recorded by the
  /// `WriterRecorder` backend's `phasm_track_bit` hook; no-op for
  /// other backends. Default no-op so most Writer impls don't need
  /// to provide it; `WriterBase<S>`'s impl forwards to its
  /// `StorageBackend::phasm_set_tag`.
  ///
  /// Callers (e.g. `encode_coeff_signs`) typically set a tag, emit
  /// one or more 50/50 symbols, then reset to `PHASM_TAG_OTHER`.
  #[inline]
  fn phasm_set_tag(&mut self, _tag: u8) {}
}

/// `StorageBackend` is an internal trait used to tie a specific `Writer`
/// implementation's storage to the generic `Writer`.  It would be
/// private, but Rust is deprecating 'private trait in a public
/// interface' support.
pub trait StorageBackend {
  /// Store partially-computed range code into given storage backend
  fn store(&mut self, fl: u16, fh: u16, nms: u16);
  /// Return bit-length of encoded stream to date
  fn stream_bits(&mut self) -> usize;
  /// Backend implementation of checkpoint to pass through Writer interface
  fn checkpoint(&mut self) -> WriterCheckpoint;
  /// Backend implementation of rollback to pass through Writer interface
  fn rollback(&mut self, _: &WriterCheckpoint);
  /// phasm-stego (fork): record a 50/50 binary emission for hybrid
  /// Pass 2 replay-with-overrides on the phasm-core side. Called from
  /// `Writer::bool` when `f == 16384` (i.e. the L(1) probability),
  /// BEFORE the corresponding `store()` call that pushes the (fl, fh,
  /// nms) tuple. Default no-op for backends that don't need replay-
  /// with-overrides support (Counter / Encoder).
  #[inline]
  fn phasm_track_bit(&mut self, _value: u16) {}

  /// phasm-stego (fork, W3.9.0): set the per-emission-site tag that
  /// the next `phasm_track_bit` call will record alongside the value.
  /// Tag is "sticky" — stays set until the next `phasm_set_tag` call.
  /// Default no-op for non-Recorder backends.
  ///
  /// Used by `encode_coeff_signs` (block_unit.rs) to distinguish:
  ///   - `PHASM_TAG_AC_COEFF_SIGN` (AC coefficient sign emissions)
  ///   - `PHASM_TAG_GOLOMB_TAIL_LSB` (golomb tail data bits)
  ///   - `PHASM_TAG_OTHER` (default for all other 50/50 emissions)
  ///
  /// Per-channel classification per `phasm-av1/docs/design/video/av1/
  /// channel-design.md` § 4. Walker filters by tag to enforce v0.3-AV1
  /// AcCoeffSign-only ship-gate (channel-design.md § 6).
  #[inline]
  fn phasm_set_tag(&mut self, _tag: u8) {}
}

// phasm-stego (W3.9.0): per-channel emission tags. Values are
// stable across phasm-rav1e + phasm-core versions (used in the
// recorder's phasm_bit_tags Vec which crosses the FFI-ish boundary).
pub const PHASM_TAG_OTHER: u8 = 0;
pub const PHASM_TAG_AC_COEFF_SIGN: u8 = 1;
pub const PHASM_TAG_GOLOMB_TAIL_LSB: u8 = 2;

/// phasm-stego (W3.10.4): per-tile recorder snapshot extracted from
/// a `WriterTee` after `encode_tile` finishes. Returned to phasm-core
/// from `encode_tile_group_with_phasm_tee` (one entry per tile in the
/// frame) so the STC plan can iterate cover positions across all
/// tiles + replay_with_overrides has the storage tuples to walk.
#[derive(Debug, Clone)]
pub struct PhasmTileRecording {
  /// `(fl, fh, nms)` tuples emitted during encode — feed to
  /// `replay_with_overrides`.
  pub storage: Vec<(u16, u16, u16)>,
  /// `(storage_index, natural_bit_value)` for every 50/50 emission —
  /// the cover position registry.
  pub bit_positions: Vec<(u32, u16)>,
  /// Per-emission channel tag (`PHASM_TAG_*`), parallel to
  /// `bit_positions`.
  pub bit_tags: Vec<u8>,
}

/// phasm-stego (W3.10.4): full-frame recording metadata returned from
/// `encode_frame_with_phasm_tee` + `Context::receive_packet_with_phasm_recording`.
/// Carries per-tile recorder data AND the byte offset where the
/// tile_group bytes start within the OBU-wrapped packet — needed by
/// phasm-core to splice stego tile bytes into the natural packet.
#[derive(Debug, Clone)]
pub struct PhasmFrameRecording {
  /// Per-tile recorder snapshot, in tile-emission order (matches
  /// `build_raw_tile_group` order).
  pub tiles: Vec<PhasmTileRecording>,
  /// Byte offset within the packet where the tile_group bytes start.
  /// `packet[..tile_group_offset]` is the OBU prefix (sequence header
  /// + frame header + tile_group_obu_header + ULEB128 size); the
  /// remainder is the raw tile bytes.
  pub tile_group_offset: usize,
  /// Length of the tile_group bytes (i.e., `packet.len() -
  /// tile_group_offset`). Phasm-core uses this for length-invariance
  /// checks on the splice (Tier 1 50/50 flips don't change tile_group
  /// length per W3.8.6).
  pub tile_group_len: usize,
}

#[derive(Debug, Clone)]
pub struct WriterBase<S> {
  /// The number of values in the current range.
  rng: u16,
  /// The number of bits of data in the current value.
  cnt: i16,
  #[cfg(feature = "desync_finder")]
  /// Debug enable flag
  debug: bool,
  /// Extra offset added to `tell()` and `tell_frac()` to approximate costs
  /// of actually coding a symbol
  fake_bits_frac: u32,
  /// Use-specific storage
  s: S,
}

#[derive(Debug, Clone)]
pub struct WriterCounter {
  /// Bits that would be shifted out to date
  bits: usize,
}

#[derive(Debug, Clone)]
pub struct WriterRecorder {
  /// Storage for tokens
  storage: Vec<(u16, u16, u16)>,
  /// Bits that would be shifted out to date
  bits: usize,
  /// phasm-stego (fork): parallel index of 50/50 binary emissions.
  /// Each entry is `(storage_index, natural_bit_value)` recorded by
  /// `phasm_track_bit` BEFORE the corresponding `store()` call (so
  /// `storage_index` is the position of the matching (fl, fh, nms)
  /// tuple in `storage`). Sorted ascending by storage_index (monotonic
  /// insert). Truncated by `rollback()` and cleared by `replay()`.
  /// Memory: ~6 bytes/entry; ~200k L(1) emissions/frame at 1080p →
  /// ~1.6 MB/frame, ~48 MB/30-frame GOP. Acceptable for mobile.
  phasm_bit_positions: Vec<(u32, u16)>,
  /// phasm-stego (fork, W3.9.0): per-position emission tag, parallel
  /// to `phasm_bit_positions`. `phasm_bit_tags[i]` = tag for the
  /// emission recorded at `phasm_bit_positions[i]`. One of
  /// `PHASM_TAG_*` constants. Default tag (when `phasm_set_tag` not
  /// called) is `PHASM_TAG_OTHER` = 0. Memory: 1 byte/entry; ~200k
  /// entries/frame at 1080p → ~200 KB/frame, ~6 MB/GOP.
  phasm_bit_tags: Vec<u8>,
  /// phasm-stego (fork, W3.9.0): sticky tag state. Set by
  /// `phasm_set_tag`; read by `phasm_track_bit` on each invocation
  /// and pushed to `phasm_bit_tags`. Stays at last-set value until
  /// next `phasm_set_tag` call.
  phasm_current_tag: u8,
}

#[derive(Debug, Clone)]
pub struct WriterEncoder {
  /// A buffer for output bytes with their associated carry flags.
  precarry: Vec<u16>,
  /// The low end of the current range.
  low: ec_window,
}

#[derive(Clone)]
pub struct WriterCheckpoint {
  /// Stream length coded/recorded to date, in the unit used by the Writer,
  /// which may be bytes or bits. This depends on the assumption
  /// that a Writer will only ever restore its own Checkpoint.
  stream_size: usize,
  /// To be defined by backend
  backend_var: usize,
  /// Saved number of values in the current range.
  rng: u16,
  /// Saved number of bits of data in the current value.
  cnt: i16,
}

/// Constructor for a counting Writer
impl WriterCounter {
  #[inline]
  pub const fn new() -> WriterBase<WriterCounter> {
    WriterBase::new(WriterCounter { bits: 0 })
  }
}

impl Default for WriterCounter {
  #[inline]
  fn default() -> Self {
    WriterCounter { bits: 0 }
  }
}

/// Constructor for a recording Writer
impl WriterRecorder {
  #[inline]
  pub const fn new() -> WriterBase<WriterRecorder> {
    WriterBase::new(WriterRecorder {
      storage: Vec::new(),
      bits: 0,
      phasm_bit_positions: Vec::new(),
      phasm_bit_tags: Vec::new(),
      phasm_current_tag: PHASM_TAG_OTHER,
    })
  }
}

impl Default for WriterRecorder {
  #[inline]
  fn default() -> Self {
    WriterRecorder {
      storage: Vec::new(),
      bits: 0,
      phasm_bit_positions: Vec::new(),
      phasm_bit_tags: Vec::new(),
      phasm_current_tag: PHASM_TAG_OTHER,
    }
  }
}

/// Constructor for a encoding Writer
impl WriterEncoder {
  #[inline]
  pub const fn new() -> WriterBase<WriterEncoder> {
    WriterBase::new(WriterEncoder { precarry: Vec::new(), low: 0 })
  }
}

impl Default for WriterEncoder {
  #[inline]
  fn default() -> Self {
    WriterEncoder { precarry: Vec::new(), low: 0 }
  }
}

/// phasm-stego (W3.10.3): combined Encoder + Recorder backend. Used
/// by phasm-core to run a single Pass 1 encode that simultaneously
/// produces:
///   - encoded bytes (via the Encoder-side precarry / low / done())
///   - position recording (via the Recorder-side storage +
///     phasm_bit_positions + phasm_bit_tags)
///
/// Mirror of `WriterRecorder` fields + `WriterEncoder` fields fused.
/// All `store()` calls update both sides atomically using the same
/// `(fl, fh, nms)` tuple → single (rng, cnt, d) computation drives
/// both the byte emission AND the storage push.
///
/// Used by phasm-core's `av1_stego_encode` orchestration so a single
/// Context API call produces both natural bytes AND the recorder
/// data needed for STC plan computation. Without WriterTee, we'd
/// need two encode passes (one for bytes, one for recording) and
/// they could diverge if rav1e's internal lookahead-derived state
/// shifts between calls.
#[derive(Debug, Clone)]
pub struct WriterTee {
  // === Encoder side ===
  precarry: Vec<u16>,
  low: ec_window,
  // === Recorder side ===
  storage: Vec<(u16, u16, u16)>,
  bits: usize,
  phasm_bit_positions: Vec<(u32, u16)>,
  phasm_bit_tags: Vec<u8>,
  phasm_current_tag: u8,
  // === Checkpoint side-state ===
  /// LIFO stack of rollback snapshots. RDO uses nested
  /// checkpoint/rollback cycles; checkpoint pushes a snapshot,
  /// rollback pops back to it. Cleared at frame boundary by the
  /// caller (or implicitly when the writer is dropped).
  pending_checkpoints: Vec<TeeRollbackState>,
}

#[derive(Debug, Clone)]
struct TeeRollbackState {
  low: ec_window,
  precarry_len: usize,
  storage_len: usize,
  bits: usize,
  phasm_bit_positions_len: usize,
  phasm_current_tag: u8,
}

impl WriterTee {
  #[inline]
  pub fn new() -> WriterBase<WriterTee> {
    WriterBase::new(WriterTee::default())
  }
}

impl Default for WriterTee {
  #[inline]
  fn default() -> Self {
    WriterTee {
      precarry: Vec::new(),
      low: 0,
      storage: Vec::new(),
      bits: 0,
      phasm_bit_positions: Vec::new(),
      phasm_bit_tags: Vec::new(),
      phasm_current_tag: PHASM_TAG_OTHER,
      pending_checkpoints: Vec::new(),
    }
  }
}

/// The Counter stores nothing we write to it, it merely counts the
/// bit usage like in an Encoder for cost analysis.
impl StorageBackend for WriterBase<WriterCounter> {
  #[inline]
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (_l, r) = self.lr_compute(fl, fh, nms);
    let d = r.leading_zeros() as usize;

    self.s.bits += d;
    self.rng = r << d;
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.bits
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_size: self.s.bits,
      backend_var: 0,
      rng: self.rng,
      // We do not use `cnt` within Counter, but setting it here allows the compiler
      // to do a 32-bit merged load/store.
      cnt: self.cnt,
    }
  }
  #[inline]
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.s.bits = checkpoint.stream_size;
  }
}

/// The Recorder does not produce a range-coded bitstream, but it
/// still tracks the range coding progress like in an Encoder, as it
/// neds to be able to report bit costs for RDO decisions.  It stores a
/// pair of mostly-computed range coding values per token recorded.
impl StorageBackend for WriterBase<WriterRecorder> {
  #[inline]
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (_l, r) = self.lr_compute(fl, fh, nms);
    let d = r.leading_zeros() as usize;

    self.s.bits += d;
    self.rng = r << d;
    self.s.storage.push((fl, fh, nms));
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.bits
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_size: self.s.bits,
      backend_var: self.s.storage.len(),
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  #[inline]
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.bits = checkpoint.stream_size;
    self.s.storage.truncate(checkpoint.backend_var);
    // phasm-stego (fork): phasm_bit_positions is sorted ascending by
    // storage_index (monotonic insert). Truncate entries pointing into
    // the rolled-back region via partition_point binary search.
    let cutoff = checkpoint.backend_var as u32;
    let pt = self
      .s
      .phasm_bit_positions
      .partition_point(|(idx, _)| *idx < cutoff);
    self.s.phasm_bit_positions.truncate(pt);
    // phasm-stego (W3.9.0): phasm_bit_tags is parallel to
    // phasm_bit_positions; truncate to the same length.
    self.s.phasm_bit_tags.truncate(pt);
  }
  /// phasm-stego (fork): record a 50/50 binary emission. Called from
  /// `Writer::bool` when `f == 16384`, BEFORE the matching `store()`
  /// call — so `storage.len()` is the index where the corresponding
  /// (fl, fh, nms) tuple will land.
  ///
  /// W3.9.0: also records the current `phasm_current_tag` value into
  /// the parallel `phasm_bit_tags` Vec.
  #[inline]
  fn phasm_track_bit(&mut self, value: u16) {
    let idx = self.s.storage.len() as u32;
    self.s.phasm_bit_positions.push((idx, value));
    self.s.phasm_bit_tags.push(self.s.phasm_current_tag);
  }
  /// phasm-stego (W3.9.0): set the per-channel emission tag. Sticky
  /// state — the next `phasm_track_bit` invocations record this tag
  /// until reset.
  #[inline]
  fn phasm_set_tag(&mut self, tag: u8) {
    self.s.phasm_current_tag = tag;
  }
}

/// An Encoder produces an actual range-coded bitstream from passed in
/// tokens.  It does not retain any information about the coded
/// tokens, only the resulting bitstream, and so it cannot be replayed
/// (only checkpointed and rolled back).
impl StorageBackend for WriterBase<WriterEncoder> {
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (l, r) = self.lr_compute(fl, fh, nms);
    let mut low = l + self.s.low;
    let mut c = self.cnt;
    let d = r.leading_zeros() as usize;
    let mut s = c + (d as i16);

    if s >= 0 {
      c += 16;
      let mut m = (1 << c) - 1;
      if s >= 8 {
        self.s.precarry.push((low >> c) as u16);
        low &= m;
        c -= 8;
        m >>= 8;
      }
      self.s.precarry.push((low >> c) as u16);
      s = c + (d as i16) - 24;
      low &= m;
    }
    self.s.low = low << d;
    self.rng = r << d;
    self.cnt = s;
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    self.s.precarry.len() * 8
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    WriterCheckpoint {
      stream_size: self.s.precarry.len(),
      backend_var: self.s.low as usize,
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.low = checkpoint.backend_var as ec_window;
    self.s.precarry.truncate(checkpoint.stream_size);
  }
}

/// phasm-stego (W3.10.3): combined Encoder + Recorder StorageBackend.
/// Every `store()` call drives both sides atomically — one
/// `lr_compute` produces the shared (l, r, d), then both the encoder
/// (precarry / low / cnt) and recorder (storage / bits) state get
/// updated. phasm_track_bit + phasm_set_tag mirror the Recorder
/// impl; rollback uses a LIFO stack of TeeRollbackState snapshots
/// indexed by backend_var (RDO checkpoint/rollback compatible).
impl StorageBackend for WriterBase<WriterTee> {
  fn store(&mut self, fl: u16, fh: u16, nms: u16) {
    let (l, r) = self.lr_compute(fl, fh, nms);
    let mut low = l + self.s.low;
    let mut c = self.cnt;
    let d = r.leading_zeros() as usize;
    let mut s = c + (d as i16);

    // Encoder side: precarry push + low / cnt update (copy of
    // WriterEncoder::store at ec.rs:265).
    if s >= 0 {
      c += 16;
      let mut m = (1 << c) - 1;
      if s >= 8 {
        self.s.precarry.push((low >> c) as u16);
        low &= m;
        c -= 8;
        m >>= 8;
      }
      self.s.precarry.push((low >> c) as u16);
      s = c + (d as i16) - 24;
      low &= m;
    }
    self.s.low = low << d;
    self.rng = r << d;
    self.cnt = s;

    // Recorder side: storage push + bits counter.
    self.s.bits += d;
    self.s.storage.push((fl, fh, nms));
  }
  #[inline]
  fn stream_bits(&mut self) -> usize {
    // Use recorder-side bit counter (more accurate than precarry * 8
    // since it tracks partial bits).
    self.s.bits
  }
  #[inline]
  fn checkpoint(&mut self) -> WriterCheckpoint {
    // Push a snapshot onto pending_checkpoints; encode the index in
    // backend_var. RDO's nested checkpoint/rollback is LIFO — when
    // rollback runs, it pops this snapshot AND any later ones.
    let idx = self.s.pending_checkpoints.len();
    self.s.pending_checkpoints.push(TeeRollbackState {
      low: self.s.low,
      precarry_len: self.s.precarry.len(),
      storage_len: self.s.storage.len(),
      bits: self.s.bits,
      phasm_bit_positions_len: self.s.phasm_bit_positions.len(),
      phasm_current_tag: self.s.phasm_current_tag,
    });
    WriterCheckpoint {
      stream_size: 0, // unused for Tee
      backend_var: idx,
      rng: self.rng,
      cnt: self.cnt,
    }
  }
  fn rollback(&mut self, checkpoint: &WriterCheckpoint) {
    let idx = checkpoint.backend_var;
    debug_assert!(
      idx < self.s.pending_checkpoints.len(),
      "WriterTee rollback to stale checkpoint idx {} (have {})",
      idx,
      self.s.pending_checkpoints.len()
    );
    let snap = self.s.pending_checkpoints[idx].clone();
    self.rng = checkpoint.rng;
    self.cnt = checkpoint.cnt;
    self.s.low = snap.low;
    self.s.precarry.truncate(snap.precarry_len);
    self.s.storage.truncate(snap.storage_len);
    self.s.bits = snap.bits;
    self.s.phasm_bit_positions.truncate(snap.phasm_bit_positions_len);
    self.s.phasm_bit_tags.truncate(snap.phasm_bit_positions_len);
    self.s.phasm_current_tag = snap.phasm_current_tag;
    // Pop this checkpoint AND any later ones (LIFO).
    self.s.pending_checkpoints.truncate(idx);
  }
  #[inline]
  fn phasm_track_bit(&mut self, value: u16) {
    let idx = self.s.storage.len() as u32;
    self.s.phasm_bit_positions.push((idx, value));
    self.s.phasm_bit_tags.push(self.s.phasm_current_tag);
  }
  #[inline]
  fn phasm_set_tag(&mut self, tag: u8) {
    self.s.phasm_current_tag = tag;
  }
}

/// phasm-stego (W3.10.3): WriterTee-specific accessors. done()
/// finalizes the encoder side (mirror of WriterBase<WriterEncoder>::done)
/// while phasm_storage / phasm_bit_positions / phasm_bit_tags expose
/// the recorder side for STC plan computation.
impl WriterBase<WriterTee> {
  /// Finalize the encoder side. Same logic as
  /// `WriterBase<WriterEncoder>::done()`.
  pub fn done(&mut self) -> Vec<u8> {
    let l = self.s.low;
    let mut c = self.cnt;
    let mut s = 10;
    let m = 0x3FFF;
    let mut e = ((l + m) & !m) | (m + 1);

    s += c;

    if s > 0 {
      let mut n = (1 << (c + 16)) - 1;

      loop {
        self.s.precarry.push((e >> (c + 16)) as u16);
        e &= n;
        s -= 8;
        c -= 8;
        n >>= 8;

        if s <= 0 {
          break;
        }
      }
    }

    let mut c = 0;
    let mut offs = self.s.precarry.len();
    let mut out = vec![0_u8; offs];
    while offs > 0 {
      offs -= 1;
      c += self.s.precarry[offs];
      out[offs] = c as u8;
      c >>= 8;
    }

    out
  }

  /// Borrow recorder-side storage tuples (parallel to bit_positions /
  /// bit_tags). For use by phasm-core's STC plan + replay machinery.
  #[inline]
  pub fn phasm_storage(&self) -> &[(u16, u16, u16)] {
    &self.s.storage
  }

  /// Borrow recorder-side 50/50 binary emission index.
  #[inline]
  pub fn phasm_bit_positions(&self) -> &[(u32, u16)] {
    &self.s.phasm_bit_positions
  }

  /// Borrow recorder-side per-emission tags.
  #[inline]
  pub fn phasm_bit_tags(&self) -> &[u8] {
    &self.s.phasm_bit_tags
  }
}

/// A few local helper functions needed by the Writer that are not
/// part of the public interface.
impl<S> WriterBase<S> {
  /// Internal constructor called by the subtypes that implement the
  /// actual encoder and Recorder. pub(crate) so generic-S encode-loop
  /// callers in `src/encoder.rs` can construct via `WriterBase::new(S::default())`.
  #[inline]
  #[cfg(not(feature = "desync_finder"))]
  pub(crate) const fn new(storage: S) -> Self {
    WriterBase { rng: 0x8000, cnt: -9, fake_bits_frac: 0, s: storage }
  }

  #[inline]
  #[cfg(feature = "desync_finder")]
  pub(crate) fn new(storage: S) -> Self {
    WriterBase {
      rng: 0x8000,
      cnt: -9,
      debug: std::env::var_os("RAV1E_DEBUG").is_some(),
      fake_bits_frac: 0,
      s: storage,
    }
  }

  /// Compute low and range values from token cdf values and local state
  const fn lr_compute(&self, fl: u16, fh: u16, nms: u16) -> (ec_window, u16) {
    let r = self.rng as u32;
    debug_assert!(32768 <= r);
    let mut u = (((r >> 8) * (fl as u32 >> EC_PROB_SHIFT))
      >> (7 - EC_PROB_SHIFT))
      + EC_MIN_PROB * nms as u32;
    if fl >= 32768 {
      u = r;
    }
    let v = (((r >> 8) * (fh as u32 >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
      + EC_MIN_PROB * (nms - 1) as u32;
    (r - u, (u - v) as u16)
  }

  /// Given the current total integer number of bits used and the current value of
  /// rng, computes the fraction number of bits used to `OD_BITRES` precision.
  /// This is used by `od_ec_enc_tell_frac()` and `od_ec_dec_tell_frac()`.
  /// `nbits_total`: The number of whole bits currently used, i.e., the value
  ///                returned by `od_ec_enc_tell()` or `od_ec_dec_tell()`.
  /// `rng`: The current value of rng from either the encoder or decoder state.
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///         rounding error is in the positive direction).
  fn frac_compute(nbits_total: u32, mut rng: u32) -> u32 {
    // To handle the non-integral number of bits still left in the encoder/decoder
    //  state, we compute the worst-case number of bits of val that must be
    //  encoded to ensure that the value is inside the range for any possible
    //  subsequent bits.
    // The computation here is independent of val itself (the decoder does not
    //  even track that value), even though the real number of bits used after
    //  od_ec_enc_done() may be 1 smaller if rng is a power of two and the
    //  corresponding trailing bits of val are all zeros.
    // If we did try to track that special case, then coding a value with a
    //  probability of 1/(1 << n) might sometimes appear to use more than n bits.
    // This may help explain the surprising result that a newly initialized
    //  encoder or decoder claims to have used 1 bit.
    let nbits = nbits_total << OD_BITRES;
    let mut l = 0;
    for _ in 0..OD_BITRES {
      rng = (rng * rng) >> 15;
      let b = rng >> 16;
      l = (l << 1) | b;
      rng >>= b;
    }
    nbits - l
  }

  const fn recenter(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
      v
    } else if v >= r {
      (v - r) << 1
    } else {
      ((r - v) << 1) - 1
    }
  }

  #[cfg(feature = "desync_finder")]
  fn print_backtrace(&self, s: u32) {
    let mut depth = 3;
    backtrace::trace(|frame| {
      let ip = frame.ip();

      depth -= 1;

      if depth == 0 {
        backtrace::resolve(ip, |symbol| {
          if let Some(name) = symbol.name() {
            println!("Writing symbol {} from {}", s, name);
          }
        });
        false
      } else {
        true
      }
    });
  }
}

/// Replay implementation specific to the Recorder
impl WriterBase<WriterRecorder> {
  /// Replays the partially-computed range tokens out of the Recorder's
  /// storage and into the passed in Writer, which may be an Encoder
  /// or another Recorder.  Clears the Recorder after replay.
  pub fn replay(&mut self, dest: &mut dyn StorageBackend) {
    // phasm-stego (W3.9.0): walk storage + phasm_bit_positions in
    // lockstep so phasm_track_bit + phasm_set_tag calls propagate
    // to the destination. Otherwise the OUTER WriterRecorder (e.g.
    // encode_tile's frame-level recorder) wouldn't see AcCoeffSign
    // / GolombTailLsb tags from per-block coefficient emissions
    // routed via the SBSQueueEntry CDEF queue (encoder.rs:3497).
    // For non-Recorder destinations (Counter / Encoder), the
    // phasm_track_bit + phasm_set_tag calls are default no-ops.
    let mut bit_iter = self
      .s
      .phasm_bit_positions
      .iter()
      .zip(self.s.phasm_bit_tags.iter())
      .peekable();
    for (i, &(fl, fh, nms)) in self.s.storage.iter().enumerate() {
      if let Some(&(&(bit_idx, bit_val), &tag)) = bit_iter.peek() {
        if bit_idx as usize == i {
          bit_iter.next();
          dest.phasm_set_tag(tag);
          dest.phasm_track_bit(bit_val);
        }
      }
      dest.store(fl, fh, nms);
    }
    self.rng = 0x8000;
    self.cnt = -9;
    self.s.storage.truncate(0);
    self.s.bits = 0;
    self.s.phasm_bit_positions.truncate(0);
    self.s.phasm_bit_tags.truncate(0);
  }

  /// phasm-stego (fork): borrow recorded storage tuples without
  /// draining the recorder. Used by phasm-core hybrid replay-with-
  /// overrides to iterate the recorded (fl, fh, nms) stream while
  /// applying L(1) overrides at the parallel `phasm_bit_positions`.
  #[inline]
  pub fn phasm_storage(&self) -> &[(u16, u16, u16)] {
    &self.s.storage
  }

  /// phasm-stego (fork): borrow recorded 50/50 binary emission index.
  /// Each entry is `(storage_index, natural_value)`. Sorted ascending
  /// by `storage_index`.
  #[inline]
  pub fn phasm_bit_positions(&self) -> &[(u32, u16)] {
    &self.s.phasm_bit_positions
  }

  /// phasm-stego (W3.9.0): borrow per-position emission tags, parallel
  /// to `phasm_bit_positions`. `phasm_bit_tags()[i]` is the
  /// `PHASM_TAG_*` value at the time `phasm_bit_positions()[i]` was
  /// recorded. Length always equals `phasm_bit_positions().len()`.
  #[inline]
  pub fn phasm_bit_tags(&self) -> &[u8] {
    &self.s.phasm_bit_tags
  }
}

/// Done implementation specific to the Encoder
impl WriterBase<WriterEncoder> {
  /// Indicates that there are no more symbols to encode.  Flushes
  /// remaining state into coding and returns a vector containing the
  /// final bitstream.
  pub fn done(&mut self) -> Vec<u8> {
    // We output the minimum number of bits that ensures that the symbols encoded
    // thus far will be decoded correctly regardless of the bits that follow.
    let l = self.s.low;
    let mut c = self.cnt;
    let mut s = 10;
    let m = 0x3FFF;
    let mut e = ((l + m) & !m) | (m + 1);

    s += c;

    if s > 0 {
      let mut n = (1 << (c + 16)) - 1;

      loop {
        self.s.precarry.push((e >> (c + 16)) as u16);
        e &= n;
        s -= 8;
        c -= 8;
        n >>= 8;

        if s <= 0 {
          break;
        }
      }
    }

    let mut c = 0;
    let mut offs = self.s.precarry.len();
    // dynamic allocation: grows during encode
    let mut out = vec![0_u8; offs];
    while offs > 0 {
      offs -= 1;
      c += self.s.precarry[offs];
      out[offs] = c as u8;
      c >>= 8;
    }

    out
  }
}

/// Generic/shared implementation for `Writer`s with `StorageBackend`s
/// (ie, `Encoder`s and `Recorder`s)
impl<S> Writer for WriterBase<S>
where
  WriterBase<S>: StorageBackend,
{
  /// Encode a single binary value.
  /// `val`: The value to encode (0 or 1).
  /// `f`: The probability that the val is one, scaled by 32768.
  fn bool(&mut self, val: bool, f: u16) {
    debug_assert!(0 < f);
    debug_assert!(f < 32768);
    // phasm-stego (fork): `bool(_, 16384)` is the 50/50 L(1)-equivalent
    // emission used by `bit()`, `literal()`, `write_golomb()`,
    // `write_subexp()`, and several direct callers (e.g. delta_lf
    // sign in src/context/block_unit.rs). Track the natural bit value
    // so phasm-core hybrid replay can apply Tier 1 overrides. Default
    // StorageBackend impl is a no-op; WriterRecorder overrides it.
    if f == 16384 {
      self.phasm_track_bit(u16::from(val));
    }
    self.symbol(u32::from(val), &[f, 0]);
  }
  /// Encode a single boolean value.
  ///
  /// - `val`: The value to encode (`false` or `true`).
  /// - `f`: The probability that the `val` is `true`, scaled by `32768`.
  fn bit(&mut self, bit: u16) {
    self.bool(bit == 1, 16384);
  }
  // fake add bits
  fn add_bits_frac(&mut self, bits_frac: u32) {
    self.fake_bits_frac += bits_frac
  }
  // phasm-stego (W3.9.0): Writer trait forwarding to StorageBackend's
  // phasm_set_tag. For WriterRecorder backend, sets the recorder's
  // tag state; for Counter/Encoder, default no-op fires via the
  // StorageBackend trait method.
  fn phasm_set_tag(&mut self, tag: u8) {
    <Self as StorageBackend>::phasm_set_tag(self, tag);
  }
  /// Encode a literal bitstring, bit by bit in MSB order, with flat
  /// probability.
  ///
  /// - 'bits': Length of bitstring
  /// - 's': Bit string to encode
  fn literal(&mut self, bits: u8, s: u32) {
    for bit in (0..bits).rev() {
      self.bit((1 & (s >> bit)) as u16);
    }
  }
  /// Encodes a symbol given a cumulative distribution function (CDF) table in Q15.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///   `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///   The values must be monotonically non-decreasing, and the last value
  ///   must be greater than 32704. There should be at most 16 values.
  ///   The lower 6 bits of the last value hold the count.
  #[inline(always)]
  fn symbol<const CDF_LEN: usize>(&mut self, s: u32, cdf: &[u16; CDF_LEN]) {
    debug_assert!(cdf[cdf.len() - 1] < (1 << EC_PROB_SHIFT));
    let s = s as usize;
    debug_assert!(s < cdf.len());
    // The above is stricter than the following overflow check: s <= cdf.len()
    let nms = cdf.len() - s;
    let fl = if s > 0 {
      // SAFETY: We asserted that s is less than the length of the cdf
      unsafe { *cdf.get_unchecked(s - 1) }
    } else {
      32768
    };
    // SAFETY: We asserted that s is less than the length of the cdf
    let fh = unsafe { *cdf.get_unchecked(s) };
    debug_assert!((fh >> EC_PROB_SHIFT) <= (fl >> EC_PROB_SHIFT));
    debug_assert!(fl <= 32768);
    self.store(fl, fh, nms as u16);
  }
  /// Encodes a symbol given a cumulative distribution function (CDF)
  /// table in Q15, then updates the CDF probabilities to reflect we've
  /// written one more symbol 's'.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///   `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///   The values must be monotonically non-decreasing, and the last value
  ///   must be greater 32704. There should be at most 16 values.
  ///   The lower 6 bits of the last value hold the count.
  fn symbol_with_update<const CDF_LEN: usize>(
    &mut self, s: u32, cdf: CDFOffset<CDF_LEN>, log: &mut CDFContextLog,
    fc: &mut CDFContext,
  ) {
    #[cfg(feature = "desync_finder")]
    {
      if self.debug {
        self.print_backtrace(s);
      }
    }
    let cdf = log.push(fc, cdf);
    self.symbol(s, cdf);

    update_cdf(cdf, s);
  }
  /// Returns approximate cost for a symbol given a cumulative
  /// distribution function (CDF) table and current write state.
  ///
  /// - `s`: The index of the symbol to encode.
  /// - `cdf`: The CDF, such that symbol s falls in the range
  ///   `[s > 0 ? cdf[s - 1] : 0, cdf[s])`.
  ///   The values must be monotonically non-decreasing, and the last value
  ///   must be greater than 32704. There should be at most 16 values.
  ///   The lower 6 bits of the last value hold the count.
  fn symbol_bits(&self, s: u32, cdf: &[u16]) -> u32 {
    let mut bits = 0;
    debug_assert!(cdf[cdf.len() - 1] < (1 << EC_PROB_SHIFT));
    debug_assert!(32768 <= self.rng);
    let rng = (self.rng >> 8) as u32;
    let fh = cdf[s as usize] as u32 >> EC_PROB_SHIFT;
    let r: u32 = if s > 0 {
      let fl = cdf[s as usize - 1] as u32 >> EC_PROB_SHIFT;
      ((rng * fl) >> (7 - EC_PROB_SHIFT)) - ((rng * fh) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB
    } else {
      let nms1 = cdf.len() as u32 - s - 1;
      self.rng as u32
        - ((rng * fh) >> (7 - EC_PROB_SHIFT))
        - nms1 * EC_MIN_PROB
    };

    // The 9 here counteracts the offset of -9 baked into cnt.  Don't include a termination bit.
    let pre = Self::frac_compute((self.cnt + 9) as u32, self.rng as u32);
    let d = r.leading_zeros() - 16;
    let mut c = self.cnt;
    let mut sh = c + (d as i16);
    if sh >= 0 {
      c += 16;
      if sh >= 8 {
        bits += 8;
        c -= 8;
      }
      bits += 8;
      sh = c + (d as i16) - 24;
    }
    // The 9 here counteracts the offset of -9 baked into cnt.  Don't include a termination bit.
    Self::frac_compute((bits + sh + 9) as u32, r << d) - pre
  }
  /// Encode a golomb to the bitstream.
  ///
  /// - 'level': passed in value to encode
  fn write_golomb(&mut self, level: u32) {
    let x = level + 1;
    let length = 32 - x.leading_zeros();

    for _ in 0..length - 1 {
      self.bit(0);
    }

    for i in (0..length).rev() {
      self.bit(((x >> i) & 0x01) as u16);
    }
  }
  /// Write a value `v` in `[0, n-1]` quasi-uniformly
  /// - `n`: size of interval
  /// - `v`: value to encode
  fn write_quniform(&mut self, n: u32, v: u32) {
    if n > 1 {
      let l = 32 - n.leading_zeros() as u8;
      let m = (1 << l) - n;
      if v < m {
        self.literal(l - 1, v);
      } else {
        self.literal(l - 1, m + ((v - m) >> 1));
        self.literal(1, (v - m) & 1);
      }
    }
  }
  /// Returns `QOD_BITRES` bits for a value `v` in `[0, n-1]` quasi-uniformly
  /// - `n`: size of interval
  /// - `v`: value to encode
  fn count_quniform(&self, n: u32, v: u32) -> u32 {
    let mut bits = 0;
    if n > 1 {
      let l = 32 - n.leading_zeros();
      let m = (1 << l) - n;
      bits += (l - 1) << OD_BITRES;
      if v >= m {
        bits += 1 << OD_BITRES;
      }
    }
    bits
  }
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite subexponential
  ///
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `v`: value to encode
  fn write_subexp(&mut self, n: u32, k: u8, v: u32) {
    let mut i = 0;
    let mut mk = 0;
    loop {
      let b = if i != 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        self.write_quniform(n - mk, v - mk);
        break;
      } else {
        let t = v >= mk + a;
        self.bool(t, 16384);
        if t {
          i += 1;
          mk += a;
        } else {
          self.literal(b, v - mk);
          break;
        }
      }
    }
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[0, n-1]` with parameter `k`
  /// as finite subexponential
  ///
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `v`: value to encode
  fn count_subexp(&self, n: u32, k: u8, v: u32) -> u32 {
    let mut i = 0;
    let mut mk = 0;
    let mut bits = 0;
    loop {
      let b = if i != 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        bits += self.count_quniform(n - mk, v - mk);
        break;
      } else {
        let t = v >= mk + a;
        bits += 1 << OD_BITRES;
        if t {
          i += 1;
          mk += a;
        } else {
          bits += (b as u32) << OD_BITRES;
          break;
        }
      }
    }
    bits
  }
  /// Write symbol `v` in `[0, n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[0, n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn write_unsigned_subexp_with_ref(&mut self, v: u32, n: u32, k: u8, r: u32) {
    if (r << 1) <= n {
      self.write_subexp(n, k, Self::recenter(r, v));
    } else {
      self.write_subexp(n, k, Self::recenter(n - 1 - r, n - 1 - v));
    }
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[0, n-1]`
  /// with parameter `k` as finite subexponential based on a
  /// reference `r` also in `[0, n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn count_unsigned_subexp_with_ref(
    &self, v: u32, n: u32, k: u8, r: u32,
  ) -> u32 {
    if (r << 1) <= n {
      self.count_subexp(n, k, Self::recenter(r, v))
    } else {
      self.count_subexp(n, k, Self::recenter(n - 1 - r, n - 1 - v))
    }
  }
  /// Write symbol `v` in `[-(n-1), n-1]` with parameter `k` as finite
  /// subexponential based on a reference `r` also in `[-(n-1), n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn write_signed_subexp_with_ref(
    &mut self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) {
    self.write_unsigned_subexp_with_ref(
      (v - low) as u32,
      (high - low) as u32,
      k,
      (r - low) as u32,
    );
  }
  /// Returns `QOD_BITRES` bits for symbol `v` in `[-(n-1), n-1]`
  /// with parameter `k` as finite subexponential based on a
  /// reference `r` also in `[-(n-1), n-1]`.
  ///
  /// - `v`: value to encode
  /// - `n`: size of interval
  /// - `k`: "parameter"
  /// - `r`: reference
  fn count_signed_subexp_with_ref(
    &self, v: i32, low: i32, high: i32, k: u8, r: i32,
  ) -> u32 {
    self.count_unsigned_subexp_with_ref(
      (v - low) as u32,
      (high - low) as u32,
      k,
      (r - low) as u32,
    )
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions.  The value
  /// will be the same whether using an `Encoder` or `Recorder`.
  ///
  /// Return: The integer number of bits.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell(&mut self) -> u32 {
    // The 10 here counteracts the offset of -9 baked into cnt, and adds 1 extra
    // bit, which we reserve for terminating the stream.
    (((self.stream_bits()) as i32) + (self.cnt as i32) + 10) as u32
      + (self.fake_bits_frac >> 8)
  }
  /// Returns the number of bits "used" by the encoded symbols so far.
  /// This same number can be computed in either the encoder or the
  /// decoder, and is suitable for making coding decisions. The value
  /// will be the same whether using an `Encoder` or `Recorder`.
  ///
  /// Return: The number of bits scaled by `2**OD_BITRES`.
  ///         This will always be slightly larger than the exact value (e.g., all
  ///          rounding error is in the positive direction).
  fn tell_frac(&mut self) -> u32 {
    Self::frac_compute(self.tell(), self.rng as u32) + self.fake_bits_frac
  }
  /// Save current point in coding/recording to a checkpoint that can
  /// be restored later.  A `WriterCheckpoint` can be generated for an
  /// `Encoder` or `Recorder`, but can only be used to rollback the `Writer`
  /// instance from which it was generated.
  fn checkpoint(&mut self) -> WriterCheckpoint {
    StorageBackend::checkpoint(self)
  }
  /// Roll back a given `Writer` to the state saved in the `WriterCheckpoint`
  ///
  /// - 'wc': Saved `Writer` state/posiiton to restore
  fn rollback(&mut self, wc: &WriterCheckpoint) {
    StorageBackend::rollback(self, wc)
  }
}

pub trait BCodeWriter {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16;
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16,
  ) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error>;
  /// # Errors
  ///
  /// - Returns `std::io::Error` if the writer cannot be written to.
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error>;
}

impl<W: io::Write> BCodeWriter for BitWriter<W, BigEndian> {
  fn recenter_nonneg(&mut self, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v around a reference r */
    if v > (r << 1) {
      v
    } else if v >= r {
      (v - r) << 1
    } else {
      ((r - v) << 1) - 1
    }
  }
  fn recenter_finite_nonneg(&mut self, n: u16, r: u16, v: u16) -> u16 {
    /* Recenters a non-negative literal v in [0, n-1] around a
    reference r also in [0, n-1] */
    if (r << 1) <= n {
      self.recenter_nonneg(r, v)
    } else {
      self.recenter_nonneg(n - 1 - r, n - 1 - v)
    }
  }
  fn write_quniform(&mut self, n: u16, v: u16) -> Result<(), std::io::Error> {
    if n > 1 {
      let l = 16 - n.leading_zeros() as u8;
      let m = (1 << l) - n;
      if v < m {
        self.write_var(l as u32 - 1, v)
      } else {
        self.write_var(l as u32 - 1, m + ((v - m) >> 1))?;
        self.write::<1, u16>((v - m) & 1)
      }
    } else {
      Ok(())
    }
  }
  fn write_subexpfin(
    &mut self, n: u16, k: u16, v: u16,
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with parameter k */
    let mut i = 0;
    let mut mk = 0;
    loop {
      let b = if i > 0 { k + i - 1 } else { k };
      let a = 1 << b;
      if n <= mk + 3 * a {
        return self.write_quniform(n - mk, v - mk);
      } else {
        let t = v >= mk + a;
        self.write_bit(t)?;
        if t {
          i += 1;
          mk += a;
        } else {
          return self.write_var(b as u32, v - mk);
        }
      }
    }
  }
  fn write_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error> {
    /* Finite subexponential code that codes a symbol v in [0, n-1] with
    parameter k based on a reference ref also in [0, n-1].
    Recenters symbol around r first and then uses a finite subexponential code. */
    let recentered_v = self.recenter_finite_nonneg(n, r as u16, v as u16);
    self.write_subexpfin(n, k, recentered_v)
  }
  fn write_s_refsubexpfin(
    &mut self, n: u16, k: u16, r: i16, v: i16,
  ) -> Result<(), std::io::Error> {
    /* Signed version of the above function */
    self.write_refsubexpfin(
      (n << 1) - 1,
      k,
      r + (n - 1) as i16,
      v + (n - 1) as i16,
    )
  }
}

pub(crate) fn cdf_to_pdf<const CDF_LEN: usize>(
  cdf: &[u16; CDF_LEN],
) -> [u16; CDF_LEN] {
  let mut pdf = [0; CDF_LEN];
  let mut z = 32768u16 >> EC_PROB_SHIFT;
  for (d, &a) in pdf.iter_mut().zip(cdf.iter()) {
    *d = z - (a >> EC_PROB_SHIFT);
    z = a >> EC_PROB_SHIFT;
  }
  pdf
}

pub(crate) mod rust {
  // Function to update the CDF for Writer calls that do so.
  #[inline]
  pub fn update_cdf<const N: usize>(cdf: &mut [u16; N], val: u32) {
    use crate::context::CDF_LEN_MAX;
    let nsymbs = cdf.len();
    let mut rate = 3 + (nsymbs >> 1).min(2);
    if let Some(count) = cdf.last_mut() {
      rate += (*count >> 4) as usize;
      *count += 1 - (*count >> 5);
    } else {
      return;
    }
    // Single loop (faster)
    for (i, v) in
      cdf[..nsymbs - 1].iter_mut().enumerate().take(CDF_LEN_MAX - 1)
    {
      if i as u32 >= val {
        *v -= *v >> rate;
      } else {
        *v += (32768 - *v) >> rate;
      }
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  const WINDOW_SIZE: i16 = 32;
  const LOTS_OF_BITS: i16 = 0x4000;

  #[derive(Debug)]
  struct Reader<'a> {
    buf: &'a [u8],
    bptr: usize,
    dif: ec_window,
    rng: u16,
    cnt: i16,
  }

  impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
      let mut r = Reader {
        buf,
        bptr: 0,
        dif: (1 << (WINDOW_SIZE - 1)) - 1,
        rng: 0x8000,
        cnt: -15,
      };
      r.refill();
      r
    }

    fn refill(&mut self) {
      let mut s = WINDOW_SIZE - 9 - (self.cnt + 15);
      while s >= 0 && self.bptr < self.buf.len() {
        assert!(s <= WINDOW_SIZE - 8);
        self.dif ^= (self.buf[self.bptr] as ec_window) << s;
        self.cnt += 8;
        s -= 8;
        self.bptr += 1;
      }
      if self.bptr >= self.buf.len() {
        self.cnt = LOTS_OF_BITS;
      }
    }

    fn normalize(&mut self, dif: ec_window, rng: u32) {
      assert!(rng <= 65536);
      let d = rng.leading_zeros() - 16;
      //let d = 16 - (32-rng.leading_zeros());
      self.cnt -= d as i16;
      /*This is equivalent to shifting in 1's instead of 0's.*/
      self.dif = ((dif + 1) << d) - 1;
      self.rng = (rng << d) as u16;
      if self.cnt < 0 {
        self.refill()
      }
    }

    fn bool(&mut self, f: u32) -> bool {
      assert!(f < 32768);
      let r = self.rng as u32;
      assert!(self.dif >> (WINDOW_SIZE - 16) < r);
      assert!(32768 <= r);
      let v = (((r >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
        + EC_MIN_PROB;
      let vw = v << (WINDOW_SIZE - 16);
      let (dif, rng, ret) = if self.dif >= vw {
        (self.dif - vw, r - v, false)
      } else {
        (self.dif, v, true)
      };
      self.normalize(dif, rng);
      ret
    }

    fn symbol(&mut self, icdf: &[u16]) -> i32 {
      let r = self.rng as u32;
      assert!(self.dif >> (WINDOW_SIZE - 16) < r);
      assert!(32768 <= r);
      let n = icdf.len() as u32 - 1;
      let c = self.dif >> (WINDOW_SIZE - 16);
      let mut v = self.rng as u32;
      let mut ret = 0i32;
      let mut u = v;
      v = ((r >> 8) * (icdf[ret as usize] as u32 >> EC_PROB_SHIFT))
        >> (7 - EC_PROB_SHIFT);
      v += EC_MIN_PROB * (n - ret as u32);
      while c < v {
        u = v;
        ret += 1;
        v = ((r >> 8) * (icdf[ret as usize] as u32 >> EC_PROB_SHIFT))
          >> (7 - EC_PROB_SHIFT);
        v += EC_MIN_PROB * (n - ret as u32);
      }
      assert!(v < u);
      assert!(u <= r);
      let new_dif = self.dif - (v << (WINDOW_SIZE - 16));
      self.normalize(new_dif, u - v);
      ret
    }
  }

  #[test]
  fn booleans() {
    let mut w = WriterEncoder::new();

    w.bool(false, 1);
    w.bool(true, 2);
    w.bool(false, 3);
    w.bool(true, 1);
    w.bool(true, 2);
    w.bool(false, 3);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert!(!r.bool(1));
    assert!(r.bool(2));
    assert!(!r.bool(3));
    assert!(r.bool(1));
    assert!(r.bool(2));
    assert!(!r.bool(3));
  }

  #[test]
  fn cdf() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(0, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 0);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }

  #[test]
  fn mixed() {
    let cdf = [7296, 3819, 1716, 0];

    let mut w = WriterEncoder::new();

    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(0, &cdf);
    w.bool(true, 2);
    w.symbol(1, &cdf);
    w.bool(true, 1);
    w.symbol(1, &cdf);
    w.bool(false, 2);
    w.symbol(1, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);
    w.symbol(2, &cdf);

    let b = w.done();

    let mut r = Reader::new(&b);

    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 0);
    assert!(r.bool(2));
    assert_eq!(r.symbol(&cdf), 1);
    assert!(r.bool(1));
    assert_eq!(r.symbol(&cdf), 1);
    assert!(!r.bool(2));
    assert_eq!(r.symbol(&cdf), 1);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
    assert_eq!(r.symbol(&cdf), 2);
  }
}
