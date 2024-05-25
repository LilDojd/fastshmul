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
