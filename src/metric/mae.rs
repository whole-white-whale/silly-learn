use ndarray::ArrayRef1;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaeError {
    #[error(
        "sample counts do not match: y_true_sample_count = {y_true_sample_count}, y_pred_sample_count = {y_pred_sample_count}"
    )]
    SampleCountsDoNotMatch {
        y_true_sample_count: usize,
        y_pred_sample_count: usize,
    },
}

pub fn mae(y_true: &ArrayRef1<f64>, y_pred: &ArrayRef1<f64>) -> Result<f64, MaeError> {
    let y_true_sample_count = y_true.len();
    let y_pred_sample_count = y_pred.len();

    if y_true_sample_count != y_pred_sample_count {
        return Err(MaeError::SampleCountsDoNotMatch {
            y_true_sample_count,
            y_pred_sample_count,
        });
    }

    Ok(1.0 / (y_true_sample_count as f64) * (y_true - y_pred).abs().sum())
}

#[cfg(test)]
mod test {
    use std::assert_matches;

    use ndarray::array;
    use ndarray_linalg::assert_aclose;

    use super::*;

    const EPSILON: f64 = 1.0e-8;

    #[test]
    fn sample_counts_do_not_match() {
        let y_true = array![0.0, 0.0, 0.0, 0.0];
        let y_pred = array![0.0, 0.0, 0.0];

        let mae_result = mae(&y_true, &y_pred);

        assert_matches!(mae_result, Err(MaeError::SampleCountsDoNotMatch { .. }));

        let y_pred = array![0.0, 0.0, 0.0, 0.0, 0.0];

        let mae_result = mae(&y_true, &y_pred);

        assert_matches!(mae_result, Err(MaeError::SampleCountsDoNotMatch { .. }));
    }

    #[test]
    fn mae_is_zero() {
        let y_true = array![0.0, 1.0];
        let y_pred = array![0.0, 1.0];

        let score = mae(&y_true, &y_pred).unwrap();

        assert_aclose!(score, 0.0, EPSILON);
    }

    #[test]
    fn mae_is_not_zero() {
        let y_true = array![1.0, 0.0];
        let y_pred = array![0.0, 1.0];

        let score = mae(&y_true, &y_pred).unwrap();

        assert_aclose!(score, 1.0, EPSILON);
    }
}
