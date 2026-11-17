# Trained model

`make train` saves `final_model.joblib` here: the fitted scikit-learn pipeline, its alert threshold
(set on the validation days), the feature list, the chosen setting and the tuned warning-sign rules.
It is rebuilt from the data, so it is not committed. Load it with `src.evaluate.load_model()`.
