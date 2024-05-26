extern crate derive_more;

pub(crate) mod errors;
pub(crate) mod parity;

pub use parity::Parity;
pub mod irreps;
pub use irreps::{Irrep, Irreps};
