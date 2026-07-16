# Churn Probability: Technical Decisions

## Purpose

Churn Probability is a small, reproducible classification project. It estimates a customer's likelihood of leaving from five details that a user can enter without consulting a full account record.

The project should demonstrate sound model evaluation and a usable prediction flow. It should not present the model as a production retention system.

## Product boundary

The Streamlit application remains the main interface. It accepts:

- tenure
- monthly charges
- contract type
- internet service
- payment method

The training pipeline and the application will use the same feature set, model artifact, threshold, and metadata. The repository will remove the current split between a full model and a separate demo model.

## Model design

The model will use a scikit-learn pipeline with:

- median imputation and standardization for numeric features
- most-frequent imputation and one-hot encoding for categorical features
- class-weighted logistic regression

Logistic regression fits the project because it produces probabilities, supports a clear decision threshold, and remains easy to inspect.

The training command will:

1. validate the dataset and target values
2. create a stratified training and holdout split
3. tune logistic-regression strength with cross-validation on the training split
4. choose a threshold from training predictions using the F2 score, which gives recall more weight than precision
5. evaluate the final pipeline once on the holdout split
6. save the model, feature schema, threshold, metrics, and evaluation charts

The holdout results will report accuracy, precision, recall, F1, F2, ROC-AUC, and average precision. The README will explain why recall matters for retention outreach and will state the cost of false positives.

## Application behavior

The app will load one versioned artifact bundle. It will validate inputs before prediction and show:

- churn probability
- the model's decision at the saved threshold
- a low, moderate, or high risk band
- holdout metrics and a short model limitation note

The app will not claim causal explanations. It may summarize the values entered by the user, but it will not label them as feature importance.

## Code structure

- `app.py`: Streamlit layout and interaction
- `src/data.py`: dataset loading, schema checks, and target normalization
- `src/modeling.py`: preprocessing, model construction, threshold selection, and metrics
- `src/train.py`: training command and artifact generation
- `src/predict.py`: command-line prediction using the saved artifact bundle
- `tests/`: data, modeling, artifact, command-line, and Streamlit smoke tests

## Failure handling

Training will stop with a specific error when the dataset, required columns, or target values are invalid. Prediction will reject missing fields, unsupported categories, negative charges, and invalid tenure. The app will show a short recovery message if artifacts are missing or incompatible.

## Verification

Automated checks will cover:

- target normalization and schema validation
- deterministic train and holdout splits
- threshold selection
- prediction input validation
- artifact loading and metadata consistency
- a Streamlit prediction flow
- formatting, linting, and a clean training run

The checked-in artifacts and reported metrics must come from the checked-in training code and dataset.
