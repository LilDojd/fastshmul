use std::fmt::Display;

use thiserror::Error;

#[derive(Error, Debug)]
#[error("Invalid input. (expected {expected}, got {found})")]
pub struct ValidationError {
    pub(crate) expected: String,
    pub(crate) found: String,
}

#[derive(Error, Debug)]
#[error("Unable to convert string {found} into {target}")]
pub struct ParseError {
    pub(crate) found: String,
    pub(crate) target: String,
}

#[derive(Error, Debug)]
pub struct UnexpectedError {
    pub(crate) msg: String,
    #[source]
    pub(crate) source: anyhow::Error,
}

impl Display for UnexpectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.msg, self.source)
    }
}
