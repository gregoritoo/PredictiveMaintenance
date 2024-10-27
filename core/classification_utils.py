import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, fmin, hp, tpe
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)


def get_machine_status(row, alpha_predicitive_strength, broken_idx):
    if any(
        (idx - int(row.name)) < alpha_predicitive_strength * 24 * 60 and (idx - int(row.name)) > 0
        for idx in broken_idx
    ):
        return "DANGER_ZONE"
    else:
        return row["machine_status"]


def mark_danger_zone(df_data, alpha_predicitive_strengh=1):
    broken_idx = df_data[df_data["machine_status"] == "BROKEN"].index.values
    df_data = df_data.copy()
    df_data["predictive_machine_status"] = df_data.apply(
        get_machine_status,
        axis=1,
        alpha_predicitive_strength=alpha_predicitive_strengh,
        broken_idx=broken_idx,
    )
    return df_data


def evaluate_model(X_test, y_test, model, inverse_custom_mapping, verbose=True):

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        raise NotImplementedError

    y_predicted = model.predict(X_test)

    balanced_acc = balanced_accuracy_score(y_test, y_predicted)

    try:
        auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_score = None

    if verbose:
        print(f"Balanced accuracy is {balanced_acc:.4f}")

        print(recall_score(y_test, y_predicted, average=None))
        if auc_score is not None:
            print(f"AUC is {auc_score:.4f}")
        else:
            print("AUC could not be calculated.")

        c_matrix = confusion_matrix(y_test, y_predicted)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=c_matrix,
            display_labels=[inverse_custom_mapping[x] for x in np.unique(y_test)],
        )
        disp.plot()
    return balanced_acc


def objective_xgb(xgb_model, params, X_train, y_train, X_test, y_test):
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    score = evaluate_model(X_test, y_test, xgb_model)
    return {"loss": -score, "status": 200}


def optimize_xgb(X_train, y_train, X_test, y_test, inverse_custom_mapping):
    params = {
        "max_depth": hp.choice("max_depth", np.arange(1, 15, dtype=int)),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1)),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "device": "cpu",
    }

    def objective_xgb_note(params):
        xgb_model = xgb.XGBClassifier(**params, tree_method="hist", enable_categorical=False)
        xgb_model.fit(X_train, y_train)
        score = evaluate_model(X_test, y_test, xgb_model, inverse_custom_mapping, verbose=False)
        return {"loss": -score, "status": STATUS_OK}

    best_params = fmin(objective_xgb_note, params, algo=tpe.suggest, max_evals=100)
    print("Best set of hyperparameters: ", best_params)
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    score = evaluate_model(X_test, y_test, xgb_model, inverse_custom_mapping, verbose=True)
    return xgb_model, best_params
