from sklearn.model_selection import GridSearchCV, StratifiedKFold
from health_disease_model_training_v2 import build_grid_search


def test_build_grid_search_returns_gridsearch():
    dummy_pipe = None
    grid = {"a": [1, 2]}
    skf = StratifiedKFold(n_splits=5)

    gs = build_grid_search(dummy_pipe, grid, skf)

    assert isinstance(gs, GridSearchCV)
    assert gs.cv == skf
    assert gs.scoring == "roc_auc"
