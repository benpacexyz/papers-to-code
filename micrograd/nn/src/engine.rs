//////////
// ### Imports

// standard
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

// python bindings
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

//////////
// ### Define Custom Classes for Micrograd

// Arc allows multiple Values to share a node in the computation graph.
// Mutex allows the node's data and gradient to be changed safely.
type NodeRef = Arc<Mutex<Node>>;

#[derive(Clone, Copy)]
enum Operation {
    None,
    Add,
    Multiply,
    Power(f64),
    Tanh,
    Relu,
}

struct Node {
    data: f64,
    grad: f64,
    previous: Vec<NodeRef>,
    operation: Operation,
    label: String,
}

// value
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Value {
    inner: NodeRef,
}

impl Value {
    pub fn new_value(data: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Node {
                data,
                grad: 0.0,
                previous: Vec::new(),
                operation: Operation::None,
                label: String::new(),
            })),
        }
    }

    fn from_operation(data: f64, previous: Vec<NodeRef>, operation: Operation) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Node {
                data,
                grad: 0.0,
                previous,
                operation,
                label: String::new(),
            })),
        }
    }

    pub fn from_python(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = value.extract::<PyRef<'_, Value>>() {
            return Ok(value.clone());
        }

        if let Ok(number) = value.extract::<f64>() {
            return Ok(Self::new_value(number));
        }

        Err(PyTypeError::new_err("expected a number or Value"))
    }

    pub fn data_value(&self) -> f64 {
        self.inner.lock().unwrap().data
    }

    pub fn grad_value(&self) -> f64 {
        self.inner.lock().unwrap().grad
    }

    pub fn set_data_value(&self, data: f64) {
        self.inner.lock().unwrap().data = data;
    }

    pub fn set_grad_value(&self, grad: f64) {
        self.inner.lock().unwrap().grad = grad;
    }

    pub fn add_value(&self, other: &Value) -> Self {
        Self::from_operation(
            self.data_value() + other.data_value(),
            vec![self.inner.clone(), other.inner.clone()],
            Operation::Add,
        )
    }

    pub fn multiply_value(&self, other: &Value) -> Self {
        Self::from_operation(
            self.data_value() * other.data_value(),
            vec![self.inner.clone(), other.inner.clone()],
            Operation::Multiply,
        )
    }

    pub fn power_value(&self, exponent: f64) -> Self {
        Self::from_operation(
            self.data_value().powf(exponent),
            vec![self.inner.clone()],
            Operation::Power(exponent),
        )
    }

    pub fn subtract_value(&self, other: &Value) -> Self {
        self.add_value(&other.multiply_value(&Self::new_value(-1.0)))
    }

    pub fn divide_value(&self, other: &Value) -> Self {
        self.multiply_value(&other.power_value(-1.0))
    }

    pub fn tanh_value(&self) -> Self {
        let x = self.data_value();
        let tanh = (2.0 * x).exp();
        let tanh = (tanh - 1.0) / (tanh + 1.0);

        Self::from_operation(tanh, vec![self.inner.clone()], Operation::Tanh)
    }

    pub fn relu_value(&self) -> Self {
        Self::from_operation(
            self.data_value().max(0.0),
            vec![self.inner.clone()],
            Operation::Relu,
        )
    }

    fn build_topology(node: &NodeRef, visited: &mut HashSet<usize>, topology: &mut Vec<NodeRef>) {
        let node_id = Arc::as_ptr(node) as usize;

        if visited.insert(node_id) {
            let previous = node.lock().unwrap().previous.clone();

            for child in previous {
                Self::build_topology(&child, visited, topology);
            }

            topology.push(node.clone());
        }
    }

    fn backward_value(&self) {
        let mut topology = Vec::new();
        let mut visited = HashSet::new();
        Self::build_topology(&self.inner, &mut visited, &mut topology);

        self.set_grad_value(1.0);

        for node in topology.into_iter().rev() {
            let (data, grad, previous, operation) = {
                let node = node.lock().unwrap();
                (node.data, node.grad, node.previous.clone(), node.operation)
            };

            match operation {
                Operation::None => {}
                Operation::Add => {
                    previous[0].lock().unwrap().grad += grad;
                    previous[1].lock().unwrap().grad += grad;
                }
                Operation::Multiply => {
                    let left_data = previous[0].lock().unwrap().data;
                    let right_data = previous[1].lock().unwrap().data;
                    previous[0].lock().unwrap().grad += right_data * grad;
                    previous[1].lock().unwrap().grad += left_data * grad;
                }
                Operation::Power(exponent) => {
                    let input_data = previous[0].lock().unwrap().data;
                    previous[0].lock().unwrap().grad +=
                        exponent * input_data.powf(exponent - 1.0) * grad;
                }
                Operation::Tanh => {
                    previous[0].lock().unwrap().grad += (1.0 - data.powi(2)) * grad;
                }
                Operation::Relu => {
                    let input_data = previous[0].lock().unwrap().data;
                    let local_grad = if input_data > 0.0 { 1.0 } else { 0.0 };
                    previous[0].lock().unwrap().grad += local_grad * grad;
                }
            }
        }
    }
}

#[pymethods]
impl Value {
    #[new]
    #[pyo3(signature = (data, label = ""))]
    fn new(data: f64, label: &str) -> Self {
        let value = Self::new_value(data);
        value.inner.lock().unwrap().label = label.to_string();
        value
    }

    #[getter]
    fn data(&self) -> f64 {
        self.data_value()
    }

    #[setter]
    fn set_data(&self, data: f64) {
        self.set_data_value(data);
    }

    #[getter]
    fn grad(&self) -> f64 {
        self.grad_value()
    }

    #[setter]
    fn set_grad(&self, grad: f64) {
        self.set_grad_value(grad);
    }

    #[getter]
    fn label(&self) -> String {
        self.inner.lock().unwrap().label.clone()
    }

    #[setter]
    fn set_label(&self, label: &str) {
        self.inner.lock().unwrap().label = label.to_string();
    }

    fn tanh(&self) -> Self {
        self.tanh_value()
    }

    fn relu(&self) -> Self {
        self.relu_value()
    }

    fn backward(&self) {
        self.backward_value();
    }

    fn __repr__(&self) -> String {
        format!("Value(data={:?})", self.data_value())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.add_value(&Self::from_python(other)?))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.multiply_value(&Self::from_python(other)?))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> Self {
        self.multiply_value(&Self::new_value(-1.0))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.subtract_value(&Self::from_python(other)?))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_python(other)?.subtract_value(self))
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(self.divide_value(&Self::from_python(other)?))
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_python(other)?.divide_value(self))
    }

    fn __pow__(&self, exponent: f64, modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err("modulo is not supported"));
        }

        Ok(self.power_value(exponent))
    }
}
