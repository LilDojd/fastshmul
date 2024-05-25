use std::{fmt::Display, str::FromStr};

use crate::errors::ParseError;
use crate::Parity;
use crate::Parity::Odd;
use anyhow::Ok as AOk;
use anyhow::Result;
use num_traits::pow;

/// Irreducible representation of O(3)
/// This struct does not contain any data, it is a structure that describe the representation.
/// It is typically used as argument of other classes of the library to define the input and output representations of
/// functions. For more information, see the `e3nn` python library.

#[derive(Debug, PartialEq)]
pub struct Irrep {
    l: u32,
    p: Parity,
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
    /// assert_eq!(irrep.l(), 6);
    /// assert_eq!(irrep.p(), Parity::Even);
    ///
    /// let irrep = Irrep::new(6, Parity::Odd);
    /// assert_eq!(irrep.p(), Parity::Odd);
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

    pub fn l(&self) -> u32 {
        self.l
    }

    pub fn p(&self) -> Parity {
        self.p
    }
    /// Create an iterator over the irreps of O(3) up to a maximum degree `lmax`
    ///
    /// # Example
    /// ```
    /// use fastshmul::Irrep;
    /// let mut iter = Irrep::iterator(Some(2));
    /// assert_eq!(iter.next().unwrap().to_string(), "0e");
    /// assert_eq!(iter.next().unwrap().to_string(), "0o");
    /// assert_eq!(iter.next().unwrap().to_string(), "1o");
    /// assert_eq!(iter.next().unwrap().to_string(), "1e");
    /// assert_eq!(iter.next().unwrap().to_string(), "2e");
    /// assert_eq!(iter.next().unwrap().to_string(), "2o");
    /// ```
    pub fn iterator(lmax: Option<u32>) -> IrrepIterator {
        IrrepIterator {
            current_l: 0,
            lmax,
            state: false,
        }
    }
}

pub struct IrrepIterator {
    current_l: u32,
    lmax: Option<u32>,
    state: bool,
}

impl Iterator for IrrepIterator {
    type Item = Irrep;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(lmax) = self.lmax {
            if self.current_l > lmax {
                return None;
            }
        }

        let parity = if !self.state {
            pow(Odd, self.current_l as usize)
        } else {
            -pow(Odd, self.current_l as usize)
        };

        let irrep = Irrep::new(self.current_l, parity);

        if !self.state {
            self.state = true;
        } else {
            self.state = false;
            self.current_l += 1;
        }

        Some(irrep)
    }
}

impl Display for Irrep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.l(), self.p())
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
    /// assert_eq!(irrep.l(), 1);
    /// assert_eq!(irrep.p(), Parity::Odd);
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
            Some('o') => AOk(Parity::from_str("o")),
            Some('e') => AOk(Parity::from_str("e")),
            Some('y') => AOk(Parity::from_str_spherical("y", l)),
            _ => {
                return Err(ParseError {
                    found: last_char.unwrap_or(' ').to_string(),
                    target: String::from("o, e, y"),
                }
                .into())
            }
        }??;

        Ok(Irrep::new(l, p))
    }
}

// 88888888888 8888888888 .d8888b. 88888888888 .d8888b.
//     888     888       d88P  Y88b    888    d88P  Y88b
//     888     888       Y88b.         888    Y88b.
//     888     8888888    "Y888b.      888     "Y888b.
//     888     888           "Y88b.    888        "Y88b.
//     888     888             "888    888          "888
//     888     888       Y88b  d88P    888    Y88b  d88P
//     888     8888888888 "Y8888P"     888     "Y8888P"

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[test]
    fn test_irrep_new() {
        let irrep = Irrep::new(6, 1);
        assert_eq!(irrep.l(), 6);
        assert_eq!(irrep.p(), Parity::Even);

        let irrep = Irrep::new(6, Parity::Odd);
        assert_eq!(irrep.p(), Parity::Odd);
    }

    #[test]
    #[should_panic]
    fn test_irrep_new_panic() {
        let _ = Irrep::new(1, 100);
    }

    #[rstest]
    #[case("1o", 1, -1)]
    #[case("1e", 1, 1)]
    #[case("1y", 1, -1)]
    #[case("2y", 2, 1)]
    #[case("3y", 3, -1)]
    #[case("4o", 4, -1)]
    #[case("0o", 0, -1)]
    fn test_irrep_try_from_valid(#[case] s: &str, #[case] l: u32, #[case] p: i8) {
        let irrep = Irrep::try_from(s).unwrap();
        assert_eq!(irrep.l(), l);
        assert_eq!(irrep.p(), p.try_into().unwrap());
    }

    #[rstest]
    #[case("babytears")]
    #[case("1")]
    #[case("1a")]
    #[case("1o1")]
    #[case("-1o")]
    #[should_panic]
    fn test_irrep_try_from_invalid(#[case] s: &str) {
        let _ = Irrep::try_from(s).unwrap();
    }

    #[rstest]
    #[case(Irrep::new(1, -1), "1o")]
    #[case(Irrep::new(7, 1), "7e")]
    #[case(Irrep::new(3, -1), "3o")]
    fn test_irrep_display(#[case] irrep: Irrep, #[case] expected: &str) {
        assert_eq!(irrep.to_string(), expected);
    }
}
