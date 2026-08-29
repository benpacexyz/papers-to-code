//////////
// ### Imports

// python bindings
use pyo3::prelude::*;
use pyo3::types::PyList;

// random numbers
use rand::Rng;

// micrograd engine
use crate::engine::Value;

fn values_from_python(x: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    x.try_iter()?
        .map(|item| Value::from_python(&item?))
        .collect()
}

//////////
// ### Define Custom Classes for Neural Network

// base behavior for neural network components
trait Module {
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        for parameter in self.parameters() {
            parameter.set_grad_value(0.0);
        }
    }
}

// neuron which represents Activation_Function(sum(x * w) + b) = scalar value
#[pyclass]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    fn build(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::rng();
        let w = (0..nin)
            .map(|_| Value::new_value(rng.random_range(-1.0..=1.0)))
            .collect();

        Self {
            w,
            b: Value::new_value(0.0),
            nonlin,
        }
    }

    fn forward(&self, x: &[Value]) -> Value {
        let mut activation = self.b.clone();

        for (weight, value) in self.w.iter().zip(x) {
            activation = activation.add_value(&weight.multiply_value(value));
        }

        if self.nonlin {
            activation.relu_value()
        } else {
            activation
        }
    }

    fn description(&self) -> String {
        let kind = if self.nonlin { "ReLU" } else { "Linear" };
        format!("{}Neuron({})", kind, self.w.len())
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut parameters = self.w.clone();
        parameters.push(self.b.clone());
        parameters
    }
}

#[pymethods]
impl Neuron {
    #[new]
    #[pyo3(signature = (nin, nonlin = true))]
    fn new(nin: usize, nonlin: bool) -> Self {
        Self::build(nin, nonlin)
    }

    fn __call__(&self, x: &Bound<'_, PyAny>) -> PyResult<Value> {
        Ok(self.forward(&values_from_python(x)?))
    }

    fn parameters(&self) -> Vec<Value> {
        Module::parameters(self)
    }

    fn zero_grad(&self) {
        Module::zero_grad(self);
    }

    fn __repr__(&self) -> String {
        self.description()
    }
}

// layer is a collection of neurons that receive the same input vector x
#[pyclass]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn build(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::build(nin, nonlin)).collect();

        Self { neurons }
    }

    fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(x))
            .collect()
    }

    fn description(&self) -> String {
        let neurons = self
            .neurons
            .iter()
            .map(Neuron::description)
            .collect::<Vec<_>>()
            .join(", ");

        format!("Layer of [{}]", neurons)
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(Module::parameters).collect()
    }
}

#[pymethods]
impl Layer {
    #[new]
    #[pyo3(signature = (nin, nout, nonlin = true))]
    fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        Self::build(nin, nout, nonlin)
    }

    fn __call__(&self, x: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
        Ok(self.forward(&values_from_python(x)?))
    }

    fn parameters(&self) -> Vec<Value> {
        Module::parameters(self)
    }

    fn zero_grad(&self) {
        Module::zero_grad(self);
    }

    fn __repr__(&self) -> String {
        self.description()
    }
}

// multi-layer perceptron class
#[pyclass]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    fn description(&self) -> String {
        let layers = self
            .layers
            .iter()
            .map(Layer::description)
            .collect::<Vec<_>>()
            .join(", ");

        format!("MLP of [{}]", layers)
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(Module::parameters).collect()
    }
}

#[pymethods]
impl MLP {
    #[new]
    fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let mut sizes = vec![nin];
        sizes.extend(&nouts);

        let mut layers = Vec::new();

        for i in 0..nouts.len() {
            let input_size = sizes[i];
            let output_size = sizes[i + 1];
            let is_last_layer = i == nouts.len() - 1;

            layers.push(Layer::build(input_size, output_size, !is_last_layer));
        }

        Self { layers }
    }

    fn __call__<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mut output = values_from_python(x)?;

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        if output.len() == 1 {
            Ok(Bound::new(py, output.remove(0))?.into_any())
        } else {
            Ok(PyList::new(py, output)?.into_any())
        }
    }

    fn parameters(&self) -> Vec<Value> {
        Module::parameters(self)
    }

    fn zero_grad(&self) {
        Module::zero_grad(self);
    }

    fn __repr__(&self) -> String {
        self.description()
    }
}
