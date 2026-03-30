from sklearn.pipeline import Pipeline
from health_disease_model_training_v2 import (
    build_logreg_pipeline,
    build_rf_pipeline
)


# def test_logistic_pipeline_structure():
#     dummy_preprocess = lambda x: x
#     pipe = build_logreg_pipeline(dummy_preprocess)

#     assert isinstance(pipe, Pipeline)
#     assert "prep" in pipe.named_steps
#     assert "clf" in pipe.named_steps


# def test_random_forest_pipeline_structure():
#     dummy_preprocess = lambda x: x
#     pipe = build_rf_pipeline(dummy_preprocess)

#     assert isinstance(pipe, Pipeline)
#     assert pipe.named_steps["clf"].random_state == 42
