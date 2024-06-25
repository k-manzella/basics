// pyfunction integration complicates stuff and I'm having trouble tracking down
// actual sources of bugs. I'm keeping these functions un-integrated for testing


fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    let result: f64 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum();
  
    result
}

fn transpose(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    for i in 0..matrix[0].len() {
        let mut row = Vec::new();
        for j in 0..matrix.len() {
            row.push(matrix[j][i]);
        }
        result.push(row);
    }
    
    result
}


fn column_slice(matrix: &Vec<Vec<f64>>, index: usize) -> Vec<f64> {
  let mut slice: Vec<f64> = vec![0.0; matrix.len()];
  for i in 0..matrix.len() {
      slice[i] = matrix[i][index];
  }

  slice
}


fn matmul(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {  
  let mut result = vec![vec![0.0; a.len()]; b[0].len()];
  for i in 0..a.len() {
      let a_row = a[i].clone();
      for j in 0..b[i].len() {
          let b_column = column_slice(&b, j);
          result[i][j] = dot_product(&a_row, &b_column)
      }
  }
  result
}

fn get_matrix_minor(m: &Vec<Vec<f64>>, i: usize, j: usize) -> Vec<Vec<f64>> {
// return matrix with rows excluding index i, columns excluding index j
  let mut rows: Vec<Vec<f64>> = m[0..i].to_vec();
  let mut following_rows: Vec<Vec<f64>> = m[i+1..].to_vec();
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



pub fn get_matrix_inverse(m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
  let determinant = get_determinant(&m);

  if m.len() == 2 {
    let mut inverse = Vec::new();
    inverse.push([m[1][1] / determinant, -1.0 * m[0][1] / determinant].to_vec());    
    inverse.push([-1.0 * m[1][0] / determinant, m[0][0] / determinant].to_vec());    
    return inverse;
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
  inverse
}

fn main() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 2.0, 6.0];
    let c = vec![7.0, 8.0, 12.0];
    let m = vec![a.clone(), b.clone(), c.clone()];

    let m_inv = get_matrix_inverse(m.clone());

    let i_test = matmul(&m, &m_inv);
    for row in i_test.iter() {
      println!("{:?}", row);
    }
}
