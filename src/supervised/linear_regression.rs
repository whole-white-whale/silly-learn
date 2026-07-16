use ndarray::{Array1, ArrayRef1, ArrayRef2};

use ndarray_linalg::Inverse;
use ndarray_linalg::error::LinalgError;

use thiserror::Error;

use crate::supervised::{SupervisedModel, TrainedSupervisedModel};

#[derive(Debug)]
pub struct TrainedLinearRegression {
    beta: Array1<f64>,
}

impl TrainedSupervisedModel<f64, f64> for TrainedLinearRegression {
    fn y(&self, x: &ArrayRef2<f64>) -> Array1<f64> {
        x.dot(&self.beta)
    }
}

#[derive(Debug, Default)]
pub struct LinearRegression {}

#[derive(Debug, Error)]
pub enum LinearRegressionError {
    #[error(
        "sample counts do not match: x_sample_count = {x_sample_count}, y_sample_count = {y_sample_count}"
    )]
    SampleCountsDoNotMatch {
        x_sample_count: usize,
        y_sample_count: usize,
    },

    #[error("sample count is zero")]
    SampleCountIsZero,

    #[error("matrix is singular")]
    MatrixIsSingular(#[from] LinalgError),
}

impl SupervisedModel<f64, f64> for LinearRegression {
    type Trained = TrainedLinearRegression;
    type Error = LinearRegressionError;

    fn train(&self, x: &ArrayRef2<f64>, y: &ArrayRef1<f64>) -> Result<Self::Trained, Self::Error> {
        let x_sample_count = x.nrows();
        let y_sample_count = y.len();

        if x_sample_count != y_sample_count {
            return Err(LinearRegressionError::SampleCountsDoNotMatch {
                x_sample_count,
                y_sample_count,
            });
        }

        if x_sample_count == 0 {
            return Err(LinearRegressionError::SampleCountIsZero);
        }

        Ok(TrainedLinearRegression {
            beta: x.t().dot(x).inv()?.dot(&x.t().dot(y)),
        })
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    use ndarray::array;
    use ndarray_linalg::assert_close_l2;

    use super::*;

    const EPSILON: f64 = 1.0e-8;

    #[test]
    fn sample_counts_do_not_match() {
        let x_train = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let y_train = array![0.0, 0.0, 0.0];

        let model = LinearRegression::default();
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LinearRegressionError::SampleCountsDoNotMatch { .. })
        );

        let y_train = array![0.0, 0.0, 0.0, 0.0, 0.0];

        let model = LinearRegression::default();
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LinearRegressionError::SampleCountsDoNotMatch { .. })
        );
    }

    #[test]
    fn matrix_is_singular() {
        let x_train = array![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let y_train = array![0.0, 0.0, 0.0, 0.0];

        let model = LinearRegression::default();
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LinearRegressionError::MatrixIsSingular { .. })
        );
    }

    #[test]
    fn identity() {
        let x_train = array![[0.0], [1.0]];
        let x_valid = array![[0.5]];

        let y_train = array![0.0, 1.0];
        let y_valid = array![0.5];

        let model = LinearRegression::default();
        let trained_model = model.train(&x_train, &y_train).unwrap();

        assert_close_l2!(&trained_model.y(&x_valid), &y_valid, EPSILON);
    }

    #[test]
    fn sum() {
        let x_train = array![[0.0, 1.0], [2.0, 3.0]];
        let x_valid = array![[4.0, 5.0]];

        let y_train = array![1.0, 5.0];
        let y_valid = array![9.0];

        let model = LinearRegression::default();
        let trained_model = model.train(&x_train, &y_train).unwrap();

        assert_close_l2!(&trained_model.y(&x_valid), &y_valid, EPSILON);
    }
}
