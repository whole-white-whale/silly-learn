use ndarray::ArrayRef1;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum R2Error {
    #[error(
        "sample counts do not match: y_true_sample_count = {y_true_sample_count}, y_pred_sample_count = {y_pred_sample_count}"
    )]
    SampleCountsDoNotMatch {
        y_true_sample_count: usize,
        y_pred_sample_count: usize,
    },

    #[error("sample count is zero")]
    SampleCountIsZero,
}

pub fn r_2(y_true: &ArrayRef1<f64>, y_pred: &ArrayRef1<f64>) -> Result<f64, R2Error> {
    let y_true_sample_count = y_true.len();
    let y_pred_sample_count = y_pred.len();

    if y_true_sample_count != y_pred_sample_count {
        return Err(R2Error::SampleCountsDoNotMatch {
            y_true_sample_count,
            y_pred_sample_count,
        });
    }

    if y_true_sample_count == 0 {
        return Err(R2Error::SampleCountIsZero);
    }

    Ok(1.0 - (y_true - y_pred).pow2().sum() / (y_true - y_true.mean().unwrap()).pow2().sum())
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

        let r_2_result = r_2(&y_true, &y_pred);

        assert_matches!(r_2_result, Err(R2Error::SampleCountsDoNotMatch { .. }));

        let y_pred = array![0.0, 0.0, 0.0, 0.0, 0.0];

        let r_2_result = r_2(&y_true, &y_pred);

        assert_matches!(r_2_result, Err(R2Error::SampleCountsDoNotMatch { .. }));
    }

    #[test]
    fn r_2_is_one() {
        let y_true = array![0.0, 1.0];
        let y_pred = array![0.0, 1.0];

        let score = r_2(&y_true, &y_pred).unwrap();

        assert_aclose!(score, 1.0, EPSILON);
    }

    #[test]
    fn r_2_is_not_one() {
        let y_true = array![1.0, 0.0];
        let y_pred = array![0.0, 1.0];

        let score = r_2(&y_true, &y_pred).unwrap();

        assert_aclose!(score, -3.0, EPSILON);
    }
}
