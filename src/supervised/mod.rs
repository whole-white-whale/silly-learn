use ndarray::{Array1, ArrayRef1, ArrayRef2};

pub mod linear_regression;

pub trait TrainedSupervisedModel<X, Y> {
    fn y(&self, x: &ArrayRef2<X>) -> Array1<Y>;
}

pub trait SupervisedModel<X, Y> {
    type Trained: TrainedSupervisedModel<X, Y>;
    type Error: std::error::Error;

    fn train(&self, x: &ArrayRef2<X>, y: &ArrayRef1<f64>) -> Result<Self::Trained, Self::Error>;
}
