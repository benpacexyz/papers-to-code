mod engine;
mod nn;

use pyo3::prelude::*;

use engine::Value;
use nn::{Layer, MLP, Neuron};

#[pymodule]
fn rust_nn(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Value>()?;
    module.add_class::<Neuron>()?;
    module.add_class::<Layer>()?;
    module.add_class::<MLP>()?;
    Ok(())
}
