use ndarray::{Array1, ArrayRef1, ArrayRef2};
use ndarray_linalg::Inverse;

use crate::supervised::{SupervisedModel, TrainedSupervisedModel};

pub struct TrainedLinearRegression {
    beta: Array1<f64>,
}

impl TrainedSupervisedModel<f64, f64> for TrainedLinearRegression {
    fn y(&self, x: &ArrayRef2<f64>) -> Array1<f64> {
        x.dot(&self.beta)
    }
}

#[derive(Default)]
pub struct LinearRegression {}

impl SupervisedModel<f64, f64> for LinearRegression {
    type Trained = TrainedLinearRegression;

    fn train(&self, x: &ArrayRef2<f64>, y: &ArrayRef1<f64>) -> Self::Trained {
        TrainedLinearRegression {
            beta: x.t().dot(x).inv().unwrap().dot(&x.t().dot(y)),
        }
    }
}

#[cfg(test)]
mod test {
    use ndarray::array;
    use ndarray_linalg::assert_close_l2;

    use super::*;

    const EPSILON: f64 = 1.0e-8;

    #[test]
    fn identity() {
        let x_train = array![[0.0], [1.0]];
        let x_valid = array![[0.5]];

        let y_train = array![0.0, 1.0];
        let y_valid = array![0.5];

        let model = LinearRegression::default();
        let trained_model = model.train(&x_train, &y_train);

        assert_close_l2!(&trained_model.y(&x_valid), &y_valid, EPSILON);
    }

    #[test]
    fn sum() {
        let x_train = array![[0.0, 1.0], [2.0, 3.0]];
        let x_valid = array![[4.0, 5.0]];

        let y_train = array![1.0, 5.0];
        let y_valid = array![9.0];

        let model = LinearRegression::default();
        let trained_model = model.train(&x_train, &y_train);

        assert_close_l2!(&trained_model.y(&x_valid), &y_valid, EPSILON);
    }
}
