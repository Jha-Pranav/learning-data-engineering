from __future__ import annotations

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra import utils
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# test pipeline


def make_prediction(input_data, config):
    _pipe_match = joblib.load(
        filename=utils.to_absolute_path(config.pipeline.pipeline01),
    )

    results = _pipe_match.predict(input_data)

    return results, _pipe_match


@hydra.main(config_name="preprocessing.yaml")
def training(config):
    current_path = utils.get_original_cwd() + "/"

    data = pd.read_csv(
        current_path + config.dataset.data,
        encoding=config.dataset.encoding,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.target.target, axis=1),
        data[config.target.target],
        test_size=0.1,
        random_state=0,
    )

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)

    pred, pipe = make_prediction(X_test, config)

    # determine mse and rmse
    print(f"test mse: {int(mean_squared_error(y_test, np.exp(pred)))}")
    print(
        f"test rmse: {int(np.sqrt(mean_squared_error(y_test, np.exp(pred))))}",
    )
    print(f"test r2: {r2_score(y_test, np.exp(pred))}")
    print(f"confusion matrix: {confusion_matrix(y_test, pred)}")

    sort_indices = np.argsort(np.abs(pipe["classifier"].coef_))[0][::-1][:5]
    print(
        "Top 5 predictors: {} with the coefficients of {}".format(
            X_train.columns[sort_indices],
            pipe["classifier"].coef_[0][sort_indices],
        ),
    )


if __name__ == "__main__":
    training()
