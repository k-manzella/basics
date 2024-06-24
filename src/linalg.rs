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


#[pyfunction]
pub fn transpose(matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let mut result = vec![vec![0.0; matrix.len()]; matrix[0].len()];

    for i in 0..matrix[0].len() {
        for j in 0..matrix.len() {
            result[i][j] = matrix[j][i];
        }
    }

    Ok(result)
}


fn column_slice(matrix: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
    let mut slice: Vec<f64> = vec![0.0; matrix.len()];
    for i in 0..matrix.len() {
        slice[i] = matrix[i][*&index];
    }

    slice
}


#[pyfunction]
pub fn matmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {  
  let mut result = vec![vec![0.0; a.len()]; b[0].len()];
  for i in 0..a.len() {
      let a_row = a[i].clone();
      for j in 0..b[i].len() {
          let b_column = column_slice(&b, j);
          result[i][j] = dot_product(a_row.clone(), b_column)?;
      }
  }

  Ok(result)
}
