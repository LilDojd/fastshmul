use crate::errors::ParseError;
use anyhow::{bail, Result};
use num_derive::ToPrimitive;
use std::{
    fmt::Display,
    ops::{Mul, Neg},
    str::FromStr,
};

use num::{One, Zero};

/// The parity of the representation. Can be either Even (1) or Odd (-1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, ToPrimitive)]
pub enum Parity {
    Even = 1,
    Odd = -1,
}

use Parity::{Even, Odd};

impl Neg for Parity {
    type Output = Parity;

    #[inline]
    fn neg(self) -> Self {
        match self {
            Even => Odd,
            Odd => Even,
        }
    }
}

impl<T: Neg<Output = T>> Mul<T> for Parity {
    type Output = T;

    #[inline]
    fn mul(self, rhs: T) -> T {
        match self {
            Even => rhs,
            Odd => -rhs,
        }
    }
}

impl One for Parity {
    fn one() -> Self {
        Even
    }
}

pub trait Signed {
    fn parity(&self) -> Option<Parity>;
}

macro_rules! impl_traits_for_primitives {
    ($($num_type: ty),*) => {
        $(
            impl From<Parity> for $num_type {
                fn from(parity: Parity) -> $num_type {
                    match parity {
                        Even => <$num_type>::one(),
                        Odd => -<$num_type>::one(),
                    }
                }
            }
            impl Signed for $num_type {

                fn parity(&self) -> Option<Parity> {
                    let zero = <$num_type>::zero();
                    match self.signum().partial_cmp(&zero) {
                        Some(std::cmp::Ordering::Greater) => Some(Even),
                        Some(std::cmp::Ordering::Less) => Some(Odd),
                        _ => None,
                    }
                }
            }

            impl TryInto<Parity> for $num_type {
                type Error = anyhow::Error;

                fn try_into(self) -> Result<Parity> {

                    match self as isize {
                        1 => Ok(Even),
                        -1 => Ok(Odd),
                        _ => bail!(ParseError {
                            found: self.to_string(),
                            target: String::from("1, -1"),
                        }),
                    }
                }
            }
        )*
    };
}

impl_traits_for_primitives!(isize, i64, i32, i16, i8, f32, f64);

impl FromStr for Parity {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "e" => Ok(Even),
            "o" => Ok(Odd),
            _ => bail!(ParseError {
                found: String::from(s),
                target: String::from("o, e"),
            }),
        }
    }
}

impl Parity {
    pub fn from_str_spherical(s: &str, order: u32) -> Result<Self> {
        match s {
            "e" => Ok(Even),
            "o" => Ok(Odd),
            "y" => Ok((-1_i8).pow(order).parity().expect("Expected valid parity")),
            _ => bail!(ParseError {
                found: String::from(s),
                target: String::from("o, e"),
            }),
        }
    }
}

impl Display for Parity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Even => write!(f, "e"),
            Odd => write!(f, "o"),
        }
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
    use num::ToPrimitive;
    use rstest::*;
    use std::str::FromStr;

    #[rstest]
    #[case("e", Even)]
    #[case("o", Odd)]
    fn test_parity_from_str(#[case] s: &str, #[case] expected: Parity) {
        let parity = Parity::from_str(s).unwrap();
        assert_eq!(parity, expected);
    }

    #[rstest]
    #[case(Even, "e")]
    #[case(Odd, "o")]
    fn test_parity_display(#[case] parity: Parity, #[case] expected: &str) {
        assert_eq!(parity.to_string(), expected);
    }

    #[test]
    fn test_parity_neg() {
        assert_eq!(-Even, Odd);
        assert_eq!(-Odd, Even);
    }

    #[rstest]
    #[case(Even, Even, Even)]
    #[case(Even, Odd, Odd)]
    #[case(Odd, Even, Odd)]
    #[case(Odd, Odd, Even)]
    fn test_parity_mul(#[case] parity: Parity, #[case] x: Parity, #[case] expected: Parity) {
        assert_eq!(parity * x, expected);
    }

    macro_rules! test_parity_mul {
        ($($num_type: ty),*) => {
            paste::item! {
                $(
                    #[rstest]
                    #[case(Even, 1 as $num_type, 1 as $num_type)]
                    #[case(Even, -1 as $num_type, -1 as $num_type)]
                    #[case(Odd, 1 as $num_type, -1 as $num_type)]
                    #[case(Odd, -1 as $num_type, 1 as $num_type)]
                    fn [< test_parity_mul_ $num_type >] (
                        #[case] parity: Parity,
                        #[case] x: $num_type,
                        #[case] expected: $num_type
                    ) { assert_eq!(parity * x, expected); }
                )*
            }
        };
    }

    test_parity_mul!(isize, i64, i32, i16, i8, f32, f64);

    macro_rules! test_parity_casting {
        ($($num_type: ty),*) => {
            paste::item! {
                $(
                    #[rstest]
                    #[case(Even, 1 as $num_type)]
                    #[case(Odd, -1 as $num_type)]
                    fn [< test_parity_into_ $num_type >] (
                        #[case] parity: Parity,
                        #[case] expected: $num_type
                    ) {
                        assert_eq!(parity as $num_type, expected); }
                )*
            }
        };
    }

    test_parity_casting!(isize, i64, i32, i16, i8);

    #[test]
    fn test_to_integers() {
        assert_eq!(Even.to_i8(), Some(1));
        assert_eq!(Odd.to_i8(), Some(-1));
        assert_eq!(1_i32, Even.into());
        assert_eq!(-1_i32, Odd.into());
    }

    #[test]
    fn test_to_float() {
        assert_eq!(Even.to_f32(), Some(1.0));
        assert_eq!(Odd.to_f32(), Some(-1.0));
        assert_eq!(1.0, Even.into());
        assert_eq!(-1.0, Odd.into());
    }

    #[rstest]
    #[case(1, Even)]
    #[case(-1, Odd)]
    #[case(100, Even)]
    #[case(-100, Odd)]
    fn test_numeric_parity(#[case] x: isize, #[case] expected: Parity) {
        assert_eq!(x.parity(), Some(expected));
    }

    #[rstest]
    #[case(std::f64::NAN, None)]
    fn test_non_numeric_parity(#[case] x: f64, #[case] expected: Option<Parity>) {
        assert_eq!(x.parity(), expected);
    }
}
