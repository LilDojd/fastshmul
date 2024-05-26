use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul},
};

use crate::errors::ParseError;
use crate::Parity;
use crate::Parity::Odd;
use anyhow::Ok as AOk;
use anyhow::{bail, Result};
use derive_more::IntoIterator;
use num::{abs, Unsigned};
use num_traits::pow;

/// Irreducible representation of `O(3)`
/// This struct does not contain any data, it is a structure that describes the representation.
/// It is typically used as argument of other classes of the library to define the input and output representations of
/// functions. For more information, see the `e3nn` python library.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct Irrep {
    pub l: u32,
    pub p: Parity,
}

/// A tuple struct to hold irrep and its multiplicity
#[doc(hidden)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct _MulIr {
    mul: u32,
    ir: Irrep,
}

/// Direct sum of irreducible representations of `O(3)`
///
/// This class does not contain any data, it is a structure that describe the representation.
/// It is typically used as argument of other classes of the library to define the input and output representations of
/// functions.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, IntoIterator)]
pub struct Irreps {
    vec: Vec<_MulIr>,
}

impl std::ops::Deref for Irreps {
    type Target = Vec<_MulIr>;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl std::ops::DerefMut for Irreps {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl Irreps {
    fn parse_irreps(input: &str) -> Result<Vec<_MulIr>> {
        input
            // Remove all whitespaces
            .split_whitespace()
            .collect::<String>()
            .split('+')
            .map(|token| {
                let parts: Vec<&str> = token.split('x').collect();
                if parts.len() == 1 {
                    return Ok(_MulIr::new(1, Irrep::try_from(parts[0])?));
                }
                let multiplicity: u32 = parts[0].parse()?;
                let irrep = Irrep::try_from(parts[1])?;
                let mulir = _MulIr::new(multiplicity, irrep);
                Ok(mulir)
            })
            .collect()
    }

    /// Multiplicity of `Irrep` in `Irreps`
    ///
    /// # Example
    /// ```
    /// use fastshmul::{Irrep, Irreps};
    /// let irreps = Irreps::try_from("2x1e + 3x2o").unwrap();
    /// assert_eq!(irreps.count(Irrep::new(1, 1)), 2);
    /// ```
    pub fn count(self, ir: Irrep) -> u32 {
        self.into_iter()
            .filter(|mulir| mulir.ir == ir)
            .map(|mulir| mulir.mul)
            .sum()
    }

    /// Check if `Irrep` is in `Irreps` with any multiplicity (zero too)
    pub fn contains(&self, ir: Irrep) -> bool {
        self.vec.iter().any(|mulir| mulir.ir == ir)
    }

    /// Simplify the `Irreps` by summing the multiplicities of the same `Irrep`
    /// Note: Returns new irrep, does not modify inplace and does not sort irreps.
    ///
    /// # Examples
    ///
    /// ```
    /// use fastshmul::Irreps;
    ///
    /// let irreps = Irreps::try_from("2x1e + 2x1e + 1x1o").unwrap();
    /// let simplified = irreps.simplify();
    /// assert_eq!(simplified.to_string(), "4x1e + 1x1o");
    /// ```
    ///
    /// Note: Equivalent representations which are separated from each other are not combined.
    ///
    /// ```
    /// use fastshmul::Irreps;
    ///
    /// let irreps = Irreps::try_from("1e + 1e + 0x1e + 0e + 1e").unwrap().simplify();
    /// assert_eq!(irreps.to_string(), "2x1e + 1x0e + 1x1e");
    /// ```
    pub fn simplify(self) -> Self {
        let iter = self.into_iter().peekable();
        let mut out: Vec<_MulIr> = Vec::new();

        // INPUT: 1e + 1e + 0x1e + 0e + 1e
        // 1. OUT: []
        //      *1e + 1e + 0x1e + 0e + 1e
        // 2. OUT: [*1x1e]
        //      1e + *1e + 0x1e + 0e + 1e
        // 3. OUT [*2x1e]
        //      1e + 1e + *0x1e + 0e + 1e
        // 4. OUT [*2x1e]
        //      1e + 1e + 0x1e + *0e + 1e
        // 5. OUT [2x1e, *1x0e]
        //      1e + 1e + 0x1e + 0e + *1e
        // 6. OUT [2x1e, 1x0e, *1x1e]
        //
        for mulir in iter {
            if let Some(last) = out.last_mut() {
                if last.ir == mulir.ir {
                    last.mul += mulir.mul;
                } else if mulir.mul > 0 {
                    out.push(mulir);
                }
            } else if mulir.mul > 0 {
                out.push(mulir);
            }
        }

        Irreps { vec: out }
    }

    /// Remove any irreps with multiplicities of zero. Returns a new `Irreps`
    ///
    /// # Example
    /// ```
    /// use fastshmul::Irreps;
    ///
    /// let irreps = Irreps::try_from("2x1e + 0x1o + 0x2e").unwrap();
    /// let simplified = irreps.remove_zero_multiplicities();
    /// assert_eq!(simplified.to_string(), "2x1e");

    pub fn remove_zero_multiplicities(self) -> Self {
        let vec = self.vec.into_iter().filter(|mulir| mulir.mul > 0).collect();
        Irreps { vec }
    }

    /// Sort the representations.
    ///
    /// # Returns
    /// A struct `SortedIrreps` containing
    /// `irreps` : `Irreps`
    /// `p` : A vector of permutation indices which will undo the sort
    /// `inv` : A vector of indices required to sort the original representations in ascending order
    ///
    /// # Examples
    ///
    /// ```
    /// use fastshmul::Irreps;
    ///
    /// assert_eq!(Irreps::try_from("1e + 0e + 1e").unwrap().sort().irreps.to_string(), "1x0e + 1x1e + 1x1e");
    ///
    /// let sorted_irreps = Irreps::try_from("2o + 1e + 0e + 1e").unwrap().sort();
    /// assert_eq!(sorted_irreps.p, vec![3, 1, 0, 2]);
    /// assert_eq!(sorted_irreps.inv, vec![2, 1, 3, 0]);
    /// ```
    pub fn sort(self) -> SortedIrreps {
        let inv = argsort(&self.vec);
        let p = inverse_permutation(&inv);
        let vec = self.vec.clone();
        let vec = inv.iter().map(|&i| vec[i]).collect();
        println!("{inv:?}");
        println!("{p:?}");

        SortedIrreps {
            irreps: Irreps { vec },
            p,
            inv,
        }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SortedIrreps {
    pub irreps: Irreps,
    pub p: Vec<usize>,
    pub inv: Vec<usize>,
}

impl From<SortedIrreps> for Irreps {
    fn from(sorted: SortedIrreps) -> Self {
        sorted.irreps
    }
}

// impl<Idx> std::ops::Index<Idx> for Irreps
// where
//     Idx: std::slice::SliceIndex<[_MulIr]>,
// {
//     // type Output = Idx::Output;
//
//     // fn index(&self, index: Idx) -> &Self::Output {
//     //     &self.vec[index]
//     // }
// }

impl From<Irrep> for Irreps {
    fn from(irrep: Irrep) -> Self {
        Irreps {
            vec: vec![_MulIr::new(1, irrep)],
        }
    }
}

impl TryFrom<&str> for Irreps {
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self> {
        match Self::parse_irreps(s) {
            Ok(vec) => Ok(Irreps { vec }),
            _ => bail!(ParseError {
                found: String::from(s),
                target: String::from("Irreps")
            }),
        }
    }
}

impl _MulIr {
    fn new(mul: u32, ir: Irrep) -> Self {
        _MulIr { mul, ir }
    }

    #[allow(dead_code)]
    fn dim(&self) -> u32 {
        self.ir.dim() * self.mul
    }
}

impl Irrep {
    /// Create a new irreducible representation `Irrep`
    ///
    /// `Irrep.p` is the parity of the representation, it can be either `Parity::Even` (1) or `Parity::Odd` (-1)
    ///
    /// # Panics
    ///
    /// This function will panic if the parity is not 1 or -1
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

    /// The dimension of the representation `2l + 1`
    pub fn dim(&self) -> u32 {
        2 * self.l + 1
    }

    /// true if l = 0 and p = 1
    pub fn is_scalar(&self) -> bool {
        self.l == 0 && self.p == Parity::Even
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

/// Iterator over the irreps of O(3)
/// The iterator will return the irreps in the order `0e, 0o, 1o, 1e, 2e, 2o, ...`
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

impl Mul for Irrep {
    type Output = Vec<Irrep>;

    /// Return `Vec<Irrep>` from the product of two `Irrep`'s
    ///
    /// # Example
    /// ```
    /// use fastshmul::Irrep;
    /// let irrep1 = Irrep::new(1, 1);
    /// let irrep2 = Irrep::new(2, -1);
    /// let irreps = irrep1 * irrep2;
    ///
    /// assert_eq!(irreps.len(), 3);
    /// assert_eq!(irreps[0].to_string(), "1o");
    /// assert_eq!(irreps[1].to_string(), "2o");
    /// assert_eq!(irreps[2].to_string(), "3o");
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        let p = self.p * rhs.p;
        let lmin = abs(self.l as i32 - rhs.l as i32) as u32;
        let lmax = self.l + rhs.l;

        (lmin..=lmax).map(|l| Irrep::new(l, p)).collect()
    }
}

/// A generic implementation of multiplication by an Unsigned
impl<T> Mul<T> for Irrep
where
    T: Unsigned + TryInto<u32>,
    T::Error: std::fmt::Debug,
{
    type Output = Irreps;

    /// Return `Irreps` from `n * Irrep` containing n `Irrep`'s'
    ///
    /// # Example
    /// ```
    /// use fastshmul::Irrep;
    ///
    /// let irrep = Irrep::new(1, 1);
    ///
    /// // Not limited to rhs
    /// let irreps = irrep * 3_u32;
    /// assert_eq!(irreps.to_string(), "3x1e");
    ///
    /// ```
    fn mul(self, rhs: T) -> Self::Output {
        // Isn't u<any> to u32 is Infallible?
        Irreps {
            vec: vec![_MulIr::new(rhs.try_into().unwrap(), self)],
        }
    }
}

// We get commutativity for free
macro_rules! left_unsigned_mul_impl {
    ($($t:ty),*) => ($(
        impl Mul<Irrep> for $t {
            type Output = Irreps;

            fn mul(self, rhs: Irrep) -> Self::Output {
                rhs * self
            }
        }
    )*)
}

left_unsigned_mul_impl!(u8, u16, u32, u64, u128, usize);

/// Add two `Irrep`'s to produce `Irreps`
impl Add for Irrep {
    type Output = Irreps;

    /// Return `Irreps` from the sum of two `Irrep`'s
    ///
    /// # Example
    ///
    /// ```
    /// use fastshmul::Irrep;
    ///
    /// let irr1 = Irrep::new(2, 1);
    ///
    /// let irreps = irr1 + irr1;
    /// assert_eq!(irreps.to_string(), "1x2e + 1x2e");
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        Irreps::from(self) + Irreps::from(rhs)
    }
}

impl Add<Irreps> for Irrep {
    type Output = Irreps;

    /// Return `Irreps` from the sum of two `Irrep`'s
    fn add(self, rhs: Irreps) -> Self::Output {
        Irreps::from(self) + rhs
    }
}

impl<T> Mul<T> for Irreps
where
    T: Unsigned + TryInto<u32>,
    T::Error: std::fmt::Debug,
{
    type Output = Irreps;

    /// Return `Irreps` from `n * Irreps`
    ///
    /// # Example
    /// ```
    /// use fastshmul::Irreps;
    ///
    /// let irreps = Irreps::try_from("2x1e").unwrap();
    ///
    /// // Not limited to rhs
    /// let irreps = irreps * 3_u32;
    /// assert_eq!(irreps.to_string(), "2x1e + 2x1e + 2x1e");
    /// ```
    fn mul(self, rhs: T) -> Self::Output {
        let rhs: u32 = rhs.try_into().unwrap();

        let vec = self.vec.repeat(rhs as usize);

        Irreps { vec }
    }
}

impl<T> Add<T> for Irreps
where
    T: Into<Irreps>,
{
    type Output = Irreps;

    /// Return `Irreps` from the sum of two `Irreps`
    fn add(self, rhs: T) -> Self::Output {
        let rhs: Irreps = rhs.into();
        let mut vec = self.vec.clone();
        // Just extend Vec, care about beauty later
        vec.extend(rhs.vec);
        Irreps { vec }
    }
}

impl Display for Irrep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.l, self.p)
    }
}

impl Display for _MulIr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.mul, self.ir)
    }
}

/// Display irreps in the form `mxlp + ... + MxXL`
///
/// # Example
/// ```
/// use fastshmul::Irreps;
/// let irrep = Irreps::try_from("100x0e + 50x1o + 8x3o").ok().unwrap();
///
/// assert_eq!(irrep.to_string(), "100x0e + 50x1o + 8x3o");
/// ```
impl Display for Irreps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.vec.iter();
        if let Some(first) = iter.next() {
            write!(f, "{}", first)?;
            for ir in iter {
                write!(f, " + {}", ir)?;
            }
        }
        Ok(())
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
            Some('o') => AOk(Parity::try_from("o")),
            Some('e') => AOk(Parity::try_from("e")),
            Some('y') => AOk(Parity::try_from_str_spherical("y", l)),
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

// 888    888          888
// 888    888          888
// 888    888          888
// 8888888888  .d88b.  888 88888b.   .d88b.  888d888 .d8888b
// 888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"   88K
// 888    888 88888888 888 888  888 88888888 888     "Y8888b.
// 888    888 Y8b.     888 888 d88P Y8b.     888          X88
// 888    888  "Y8888  888 88888P"   "Y8888  888      88888P'
//                         888
//                         888
//                         888

fn argsort<T: Ord>(data: &[T]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &data[i]);
    indices
}

fn inverse_permutation(p: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; p.len()];
    for (i, &x) in p.iter().enumerate() {
        inv[x] = i;
    }
    inv
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
        assert_eq!(irrep.l, 6);
        assert_eq!(irrep.p, Parity::Even);

        let irrep = Irrep::new(6, Parity::Odd);
        assert_eq!(irrep.p, Parity::Odd);
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
        assert_eq!(irrep.l, l);
        assert_eq!(irrep.p, p.try_into().unwrap());
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

    #[fixture]
    fn irrep_tuple() -> (Irrep, Irrep) {
        (Irrep::new(1, -1), Irrep::new(2, 1))
    }

    #[rstest]
    fn test_irrep_mul_self(irrep_tuple: (Irrep, Irrep)) {
        let (ir1, ir2) = irrep_tuple;
        let irreps = ir1 * ir2;
        assert_eq!(irreps.len(), 3);
        assert_eq!(irreps[0].to_string(), "1o");
        assert_eq!(irreps[1].to_string(), "2o");
        assert_eq!(irreps[2].to_string(), "3o");
    }

    #[test]
    fn test_irrep_mul_integer() {
        let irrep = Irrep::new(1, -1);
        let irreps = irrep * 3_u32;
        assert_eq!(irreps.vec.len(), 1);
        assert_eq!(irreps.vec[0].mul, 3);
        assert_eq!(irreps.vec[0].ir.l, 1);
        assert_eq!(irreps.vec[0].ir.p, Parity::Odd);
        let other_irreps = 3_u32 * irrep;
        assert_eq!(irreps, other_irreps);
    }

    #[rstest]
    fn test_irrep_add_self(irrep_tuple: (Irrep, Irrep)) {
        let (ir1, ir2) = irrep_tuple;
        let irreps = ir1 + ir2;
        assert_eq!(irreps.vec.len(), 2);
        assert_eq!(irreps.vec[0].mul, 1);
        assert_eq!(irreps.vec[0].ir.l, 1);
        assert_eq!(irreps.vec[0].ir.p, Parity::Odd);
        assert_eq!(irreps.vec[1].mul, 1);
        assert_eq!(irreps.vec[1].ir.l, 2);
        assert_eq!(irreps.vec[1].ir.p, Parity::Even);
        let other_irreps = ir2 + ir1;
        assert_eq!(
            irreps.vec.into_iter().rev().collect::<Vec<_>>(),
            other_irreps.vec
        );
    }

    #[rstest]
    fn test_irrep_add_irreps(irrep_tuple: (Irrep, Irrep)) {
        let (ir1, ir2) = irrep_tuple;
        let irreps = Irreps::from(ir1) + ir2;
        assert_eq!(irreps.vec.len(), 2);
        assert_eq!(irreps.vec[0].mul, 1);
        assert_eq!(irreps.vec[0].ir.l, 1);
        assert_eq!(irreps.vec[0].ir.p, Parity::Odd);
        assert_eq!(irreps.vec[1].mul, 1);
        assert_eq!(irreps.vec[1].ir.l, 2);
        assert_eq!(irreps.vec[1].ir.p, Parity::Even);
        let other_irreps = ir2 + Irreps::from(ir1);
        assert_eq!(
            irreps.vec.into_iter().rev().collect::<Vec<_>>(),
            other_irreps.vec
        );
    }

    #[rstest]
    #[case("100x0e + 50x1o + 8x3o", 3)]
    #[case("1x0e+12x1o+2x3o", 3)]
    #[case("2x0o", 1)]
    #[case("5e", 1)]
    #[case("0x0e", 1)]
    fn test_irreps_from_str(#[case] s: &str, #[case] expected_len: usize) {
        let irreps = Irreps::try_from(s).unwrap();
        assert_eq!(irreps.vec.len(), expected_len);
    }

    #[rstest]
    #[case("aboba")]
    fn test_irreps_from_str_errors(#[case] s: &str) {
        let irreps = Irreps::try_from(s);
        assert!(irreps.is_err());
    }

    #[rstest]
    #[case("100x0e + 50x1o + 8x3o", vec!((100, 0, Parity::Even), (50, 1, Parity::Odd), (8, 3, Parity::Odd)))]
    #[case("1x0e+12x1o+2x3o", vec!((1, 0, Parity::Even), (12, 1, Parity::Odd), (2, 3, Parity::Odd)))]
    #[case("2x0o", vec!((2, 0, Parity::Odd)))]
    #[case("5e", vec!((1, 5, Parity::Even)))]
    fn test_irreps_parse_irreps(#[case] s: &str, #[case] expected: Vec<(u32, u32, Parity)>) {
        let irreps = Irreps::parse_irreps(s).unwrap();
        assert_eq!(irreps.len(), expected.len());
        for (i, (m, l, p)) in expected.iter().enumerate() {
            assert_eq!(irreps[i].mul, *m);
            assert_eq!(irreps[i].ir.l, *l);
            assert_eq!(irreps[i].ir.p, *p);
        }
    }

    #[rstest]
    #[case(Irrep::new(1, 1), 1, Parity::Even)]
    #[case(Irrep::new(3, -1), 3, Parity::Odd)]
    fn test_irreps_from_irrep(#[case] irrep: Irrep, #[case] l: u32, #[case] p: Parity) {
        let irreps = Irreps::from(irrep);
        assert_eq!(irreps.vec.len(), 1);
        assert_eq!(irreps.vec[0].mul, 1);
        assert_eq!(irreps.vec[0].ir.l, l);
        assert_eq!(irreps.vec[0].ir.p, p);
    }
}
