use crate::errors::{ParseError, ValidationError};
use anyhow::Ok as AOk;
use anyhow::{bail, Result};
use num::FromPrimitive;

/// Irreducible representation of O(3)
/// This struct does not contain any data, it is a structure that describe the representation.
/// It is typically used as argument of other classes of the library to define the input and output representations of
/// functions. For more information, see the `e3nn` python library.

#[derive(Debug, PartialEq)]
pub struct Irrep {
    pub l: u32,
    pub p: Parity,
}

impl Irrep {
    /// Create a new irreducible representation `Irrep`
    ///
    /// `Irrep.p` is the parity of the representation, it can be either `Parity::Even` (1) or `Parity::Odd` (-1)
    ///
    /// # Errors
    ///
    /// This function will return an error if the parity is not 1 or -1
    ///
    /// # Examples
    /// ```
    /// use fastshmul::{Irrep, Parity};
    ///
    /// let irrep = Irrep::new(6, 1);
    ///
    /// assert_eq!(irrep.l, 6);
    /// assert_eq!(irrep.p, Parity::Even);
    ///
    /// let irrep = Irrep::new(6, Parity::Odd);
    /// assert_eq!(irrep.p, Parity::Odd);
    /// ```
    ///
    /// Trying to pass numeric values other than 1 or -1 will panic:
    ///
    /// ```should_panic
    /// use fastshmul::Irrep;
    /// let irrep = Irrep::new(1, 100);
    /// ```
    pub fn new<T: TryInto<Parity>>(l: u32, p: T) -> Self
    where
        T::Error: Into<anyhow::Error> + std::fmt::Debug,
    {
        let p = p.try_into().unwrap();
        Irrep { l, p }
    }
}

impl TryFrom<&str> for Irrep {
    type Error = anyhow::Error;

    /// Create a new `Result<Irrep>` from `&str`
    ///
    /// We follow the e3nn convention in which irrep strings contain irrep degree l = 0,1,...
    /// and the parity of the representation `o` for odd, `e` for even and `y` for spherical
    /// harmonics parity (-1^l).
    ///
    /// # Errors
    /// Will produce an error if the string does not follow the e3nn convention.
    ///
    /// # Example
    /// ```
    /// use fastshmul::{Irrep, Parity};
    ///
    /// // This will product a vector irrep (l=1, odd)
    /// let irrep = Irrep::try_from("1o").unwrap();
    ///
    /// assert_eq!(irrep.l, 1);
    /// assert_eq!(irrep.p, Parity::Odd);
    ///
    /// let invalid_irrep = Irrep::try_from("babytears");
    /// assert!(invalid_irrep.is_err());
    /// ```
    fn try_from(s: &str) -> Result<Self> {
        let name = s.trim();
        // l should be the first number
        let order_l = name
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>();

        let l = order_l.parse::<u32>()?;
        let last_char = name.chars().last();
        let p = match last_char {
            Some('o') => AOk(-1),
            Some('e') => AOk(1),
            Some('y') => AOk((1_i8).pow(l)),
            _ => {
                return Err(ParseError {
                    found: last_char.unwrap_or(' ').to_string(),
                    target: String::from("o, e, y"),
                }
                .into())
            }
        }?;

        Ok(Irrep::new(l, p))
    }
}

/// The parity of the representation. Can be either 1 or -1
#[derive(Debug, num_derive::FromPrimitive, num_derive::ToPrimitive, PartialEq)]
pub enum Parity {
    Even = 1,
    Odd = -1,
}

impl TryFrom<i8> for Parity {
    type Error = anyhow::Error;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        if let Some(v) = FromPrimitive::from_i8(value) {
            Ok(v)
        } else {
            bail!(ValidationError {
                expected: String::from("{1, -1}"),
                found: value.to_string(),
            })
        }
    }
}
