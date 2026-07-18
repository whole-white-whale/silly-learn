use ndarray::{Array1, Array2, ArrayRef1, ArrayRef2, Axis, concatenate};

use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{Inverse, Norm};

use thiserror::Error;

use crate::supervised::{SupervisedModel, TrainedSupervisedModel};

fn sigmoid(x: &ArrayRef2<f64>, beta: &ArrayRef1<f64>) -> Array1<f64> {
    1.0 / (1.0 + (-x.dot(beta)).exp())
}

#[derive(Debug)]
pub struct TrainedLogisticRegression {
    beta: Array1<f64>,
}

impl TrainedSupervisedModel<f64, bool> for TrainedLogisticRegression {
    fn y(&self, x: &ArrayRef2<f64>) -> Array1<bool> {
        let x = concatenate(
            Axis(1),
            &[Array2::from_elem((x.nrows(), 1), 1.0).view(), x.view()],
        )
        .unwrap();

        sigmoid(&x, &self.beta).mapv(|p| p > 0.5)
    }
}

#[derive(Debug)]
pub struct LogisticRegression {
    epsilon: f64,
    max_iteration_count: usize,
}

impl LogisticRegression {
    pub fn new(epsilon: f64, max_iteration_count: usize) -> Self {
        Self {
            epsilon,
            max_iteration_count,
        }
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            epsilon: 1.0e-8,
            max_iteration_count: 100,
        }
    }
}

#[derive(Debug, Error)]
pub enum LogisticRegressionError {
    #[error("epsilon is negative: epsilon = {epsilon}")]
    EpsilonIsNegative { epsilon: f64 },

    #[error(
        "sample counts do not match: x_sample_count = {x_sample_count}, y_sample_count = {y_sample_count}"
    )]
    SampleCountsDoNotMatch {
        x_sample_count: usize,
        y_sample_count: usize,
    },

    #[error("sample count is zero")]
    SampleCountIsZero,

    #[error("hessian is singular")]
    HessianIsSingular(#[from] LinalgError),

    #[error(
        "optimization has not converged: epsilon = {epsilon}, max_iteration_count = {max_iteration_count}"
    )]
    OptimizationHasNotConverged {
        epsilon: f64,
        max_iteration_count: usize,
    },
}

impl SupervisedModel<f64, bool> for LogisticRegression {
    type Trained = TrainedLogisticRegression;
    type Error = LogisticRegressionError;

    fn train(&self, x: &ArrayRef2<f64>, y: &ArrayRef1<bool>) -> Result<Self::Trained, Self::Error> {
        if self.epsilon < 0.0 {
            return Err(LogisticRegressionError::EpsilonIsNegative {
                epsilon: self.epsilon,
            });
        }

        let x_sample_count = x.nrows();
        let y_sample_count = y.len();

        if x_sample_count != y_sample_count {
            return Err(LogisticRegressionError::SampleCountsDoNotMatch {
                x_sample_count,
                y_sample_count,
            });
        }

        if x_sample_count == 0 {
            return Err(LogisticRegressionError::SampleCountIsZero);
        }

        let x = concatenate(
            Axis(1),
            &[Array2::from_elem((x_sample_count, 1), 1.0).view(), x.view()],
        )
        .unwrap();

        let mut beta = Array1::zeros(x.ncols());

        for _ in 0..self.max_iteration_count {
            let p = sigmoid(&x, &beta);
            let g = x.t().dot(&(y.map(|&y| if y { 1.0 } else { 0.0 }) - &p));

            if g.norm() < self.epsilon {
                return Ok(TrainedLogisticRegression { beta });
            }

            let w = Array2::from_diag(&p.map(|&p| p * (1.0 - p)));
            let h = x.t().dot(&w).dot(&x);

            beta = beta + h.inv()?.dot(&g);
        }

        Err(LogisticRegressionError::OptimizationHasNotConverged {
            epsilon: self.epsilon,
            max_iteration_count: self.max_iteration_count,
        })
    }
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    use ndarray::array;

    use super::*;

    #[test]
    fn epsilon_is_negative() {
        let x_train = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let y_train = array![false, false, false, false];

        let model = LogisticRegression::new(-1.0, 100);
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LogisticRegressionError::EpsilonIsNegative { .. })
        );
    }

    #[test]
    fn sample_counts_do_not_match() {
        let x_train = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let y_train = array![false, false, false];

        let model = LogisticRegression::default();
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LogisticRegressionError::SampleCountsDoNotMatch { .. })
        );

        let y_train = array![false, false, false, false, false];

        let model = LogisticRegression::default();
        let train_result = model.train(&x_train, &y_train);

        assert_matches!(
            train_result,
            Err(LogisticRegressionError::SampleCountsDoNotMatch { .. })
        );
    }

    #[test]
    fn is_always_false() {
        let x_train = array![[0.0], [1.0]];
        let x_valid = array![[0.5]];

        let y_train = array![false, false];
        let y_valid = array![false];

        let model = LogisticRegression::default();
        let trained_model = model.train(&x_train, &y_train).unwrap();

        assert_eq!(&trained_model.y(&x_valid), &y_valid);
    }

    #[test]
    fn is_more_than_half() {
        let x_train = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
        let x_valid = array![[0.3], [0.7]];

        let y_train = array![false, false, false, true, true, true];
        let y_valid = array![false, true];

        let model = LogisticRegression::default();
        let trained_model = model.train(&x_train, &y_train).unwrap();

        assert_eq!(&trained_model.y(&x_valid), &y_valid);
    }
}
