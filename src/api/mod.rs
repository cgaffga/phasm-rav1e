// Copyright (c) 2018-2020, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

/// Channel-based encoder
#[cfg(all(feature = "channel-api", feature = "unstable"))]
pub mod channel;
/// Color model information
pub mod color;
/// Encoder Configuration
pub mod config;
/// Encoder Context
pub mod context;
/// Internal implementation
pub(crate) mod internal;
/// Lookahead-specific methods
pub(crate) mod lookahead;

mod util;

#[cfg(test)]
mod test;

#[cfg(all(feature = "channel-api", feature = "unstable"))]
pub use channel::*;
pub use color::*;
pub use config::*;
pub use context::*;
pub(crate) use internal::*;
// phasm-stego (W3.8.3): selectively re-export InterConfig so the
// `crate::phasm_stego` mod can route phasm-core callers to it.
// TODO(v0.4+): drop when the Option 1 refactor lands.
pub use internal::InterConfig as PhasmInterConfig;
pub use util::*;
