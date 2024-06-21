use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyfunction]
pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {

    let result: f64 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();
    Ok(result)
}


#[pyfunction]
pub fn magnitude(a: Vec<f64>) -> PyResult<f64> {
    let sum_squares: f64 = a.iter()
        .map(|&x| x * x)
        .sum();
    let result: f64 = sum_squares.sqrt();
    Ok(result)
}


#[pyfunction]
pub fn normalize(a: Vec<f64>) -> PyResult<Vec<f64>> {
    let mag: f64 = magnitude(a.clone())?;
    if mag == 0.0 {
        return Err(PyErr::new::<PyValueError, _>("Cannot normalize vector with magnitude 0"));
    }

    let result: Vec<f64> = a.iter().map(|&x| x / mag).collect();
    Ok(result)
}

 
// #[pyfunction]
// pub fn matmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    // let a_m: i32 = a.len();
    // let a_n: i32 = a.iter().skip(1).len();
    // let b_n: i32 = b.len();
    // let b_p: i32 = b.iter().skip(1).len();
    // if a_n != b_n {
        // return Err(PyErr::new::<PyValueError, _>("Matrix dimension mismatch"));
    // }
// 
    // 
    // Ok()
// }


// #[pyfunction]
// pub fn mat_inverse(a: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
//     Ok()
// }
