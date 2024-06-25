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


fn get_matrix_minor(m: &Vec<Vec<f64>>, i: usize, j: usize) -> Vec<Vec<f64>> {
  let mut rows: Vec<Vec<f64>> = m[0..i].to_vec();
  let mut following_rows: Vec<Vec<f64>> = m[i+1..].to_vec();
  rows.append(&mut following_rows);
  let mut minor = Vec::new();
  for i in rows.iter_mut() {
    i.remove(j);
    minor.push(i.to_vec());
  }
  minor
}


fn get_determinant(m: &Vec<Vec<f64>>) -> f64 {
  if m.len() == 2 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
  }
  
  let mut determinant: f64 = 0.0;
  for c in 0..m.len() {
    let minor = get_matrix_minor(&m, 0, c);
    let addition = ((-1.0_f64).powi((c) as i32)) * m[0][c] * get_determinant(&minor);
    determinant += addition;
  }
  determinant
}


#[pyfunction]
pub fn get_matrix_inverse(m: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
  let determinant = get_determinant(&m);
  if determinant == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Matrix is singular and cannot be inverted."));
  }
  
  if m.len() == 2 {
    let mut inverse = Vec::new();
    inverse.push([m[1][1] / determinant, -1.0 * m[0][1] / determinant].to_vec());
    inverse.push([-1.0 * m[1][0] / determinant, m[0][0] / determinant].to_vec());
    return Ok(inverse);
  }

  let mut cofactors = Vec::new();
  for r in 0..m.len() {
    let mut cofactor_row = Vec::new();
    for c in 0..m.len() {
      let minor = get_matrix_minor(&m, r, c);
      let det_minor = get_determinant(&minor);
      cofactor_row.push(((-1.0_f64).powi((r + c) as i32)) * det_minor);
    }
  cofactors.push(cofactor_row.to_vec());
  }

  let cofactors_t = transpose(cofactors);
  let mut inverse = Vec::new();
  for row in cofactors_t.iter() {
    let mut inv_row = Vec::new();
    for &element in row.iter() {
      inv_row.push(element / determinant);
    }
    inverse.push(inv_row);
  }
  Ok(inverse)
}
