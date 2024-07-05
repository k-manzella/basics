use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::Rng;

#[pyfunction]
pub fn train_test_split(
    x: Vec<Vec<f64>>, 
    y: Vec<f64>, 
    test_size: f64
) -> PyResult<(
        Vec<Vec<f64>>, 
        Vec<Vec<f64>>, 
        Vec<f64>, 
        Vec<f64>
)> {
    if x.len() != y.len() {
        return Err(PyErr::new::<PyValueError, _>("Dimension mismatch"));
    }
    let mut rng = rand::thread_rng();
    let n: usize = y.len();
    let test_n: usize = ((n as f64) * test_size).floor() as usize;
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    let mut x_train = x.clone();
    let mut y_train = y.clone();
    for i in 0..test_n {
        let range_len = &n - i;
        let idx = rng.gen_range(0..range_len);
        x_test.push(x_train.remove(idx));
        y_test.push(y_train.remove(idx));
    }
    return Ok((x_train, x_test, y_train, y_test))
}
