mod linalg;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn is_prime(num: u32) -> bool {
    match num {
        0 | 1 => false,
        _ => {
            let limit = (num as f32).sqrt() as u32;

            (2..=limit).any(|i| num % i == 0) == false
        }
    }
}


/// A Python module implemented in Rust.
#[pymodule]
fn basics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::get_matrix_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(linalg::r_squared, m)?)?;
    Ok(())
}


/// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test_false() {
        assert_eq!(is_prime(0), false);
        assert_eq!(is_prime(1), false);
        assert_eq!(is_prime(12), false);
    }

    #[test]
    fn simple_test_true() {
        assert_eq!(is_prime(2), true);
        assert_eq!(is_prime(3), false);
        assert_eq!(is_prime(4), true);
    }
}

