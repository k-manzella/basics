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
    let mut result = Vec::new();
    for i in 0..matrix[0].len() {
        let mut row = Vec::new();
        for j in 0..matrix.len() {
            row.push(matrix[j][i]);
        }
        result.push(row);
    }
    Ok(result)
}


fn column_slice(matrix: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
  let mut slice: Vec<f64> = vec![0.0; matrix.len()];
  for i in 0..matrix.len() {
      slice[i] = matrix[i][index];
  }
  slice
}


fn mat_vec_multiply(m: Vec<Vec<f64>>, v: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let column_vector = column_slice(&v, 0);
  let mut result_elements = vec![0.0; m.len()];
  for i in 0..m.len() {
        let m_row = m[i].clone();
        result_elements[i] = dot_product(m_row, column_vector.clone()).unwrap();    }
    // make result a nested vector to satisfy typing (for now)
  let result = vec![result_elements];
  return result
}


#[pyfunction]
pub fn matmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {  
  if (a[0].len() != b.len()) && (b.len() != 1) {
    return Err(PyErr::new::<PyValueError, _>("Dimension mismatch"));
  }

  // special case for multiplying matrix by vector
  // vector still has to be turned into vector of vectors in order to be passed
  if b.len() == 1 {
    let b_t: Vec<Vec<f64>> = transpose(b).unwrap();
    let result = mat_vec_multiply(a, b_t);
    return Ok(result)  
  } else if b[0].len() == 1 {
    let result = mat_vec_multiply(a, b);
    return Ok(result)
  } 

  let mut result = Vec::new();
  for i in 0..a.len() {
      let a_row = a[i].clone();
      let mut result_row = Vec::new();
      for j in 0..b[0].len() {
          let b_col = column_slice(&b, j);
          result_row.push(dot_product(a_row.clone(), b_col.clone()).unwrap());
      }
      result.push(result_row);
  }
  Ok(result)
}


fn get_matrix_minor(m: &Vec<Vec<f64>>, i: usize, j: usize) -> Vec<Vec<f64>> {
// return matrix with rows excluding index i, columns excluding index j
  let mut rows: Vec<Vec<f64>> = m[0..i].to_vec();
  let mut following_rows = m[i + 1..].to_vec(); 
  rows.append(&mut following_rows);
  let mut minor = vec![];
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

  let cofactors_t = transpose(cofactors).expect("Failed to transpose cofactors during inverse");
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


#[pyfunction]
pub fn r_squared(y: Vec<f64>, y_hat: Vec<f64>) -> PyResult<f64> {
  let y_mean: f64 = y.iter().sum::<f64>() / (y.len() as f64);
  let sst: f64 = y.iter()
    .map(|y| (y - y_mean).powi(2) )
    .sum();

  let ssr: f64 = y.iter()
    .zip(y_hat.iter())
    .map(|(y_i, y_h)| (y_i - y_h).powi(2) )
    .sum::<f64>();

  let result = 1.0 - (ssr / sst);
  Ok(result)
} 
