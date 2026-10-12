# Census Income Classifier Model Card

## Model Details

- **Model:** Scikit-learn `RandomForestClassifier`
- **Version:** Initial project model, trained with random seed 42
- **Configuration:** 100 trees, maximum depth 10, minimum 2 samples per leaf
- **Preprocessing:** One-hot encoding for categorical features; continuous features are passed through without scaling. Unknown categories are ignored by the encoder.
- **Artifacts:** The trained classifier and fitted encoder are stored in `model/`.

## Intended Use

This model demonstrates a reproducible classification workflow for the supplied Census income dataset. It predicts whether a row's recorded salary category is `>50K` or `<=50K`. It is intended for coursework and experimentation, not for decisions about real people.

## Training Data

The model uses `data/census.csv` (32,561 rows and 15 columns). The target is `salary`; the remaining columns are model features. The data is split into 80% training and 20% evaluation partitions with `random_state=42` and stratification by salary. The training partition contains 26,048 rows.

Categorical features are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`. The other feature columns are treated as continuous or numeric inputs.

## Evaluation Data

The held-out evaluation partition contains 6,513 rows and is not used to fit the classifier or encoder. It is used for the overall metrics and the categorical slice results in `slice_output.txt`.

## Metrics

On the held-out partition:

- **Precision:** 0.8074
- **Recall:** 0.5427
- **F1 score:** 0.6491

The slice report includes precision, recall, and F1 for each value of each categorical feature. Metrics for very small slices are unstable; for example, a slice with one row can show a perfect score without providing meaningful evidence of performance.

## Ethical Considerations

The target represents historical salary categories and is associated with attributes such as race, sex, age, and occupation. Model predictions may reproduce or amplify historical inequities. The model should not be used for hiring, lending, benefits, compensation, or any other consequential decision about an individual.

## Caveats and Recommendations

- Performance is measured on one random, stratified holdout from the supplied dataset. It does not establish performance on current populations or other datasets.
- The evaluation results show a recall of 0.5427, so the classifier misses a substantial share of examples whose recorded salary is above 50K.
- Review slice metrics and sample counts before drawing conclusions about subgroup performance; improve evaluation for slices with limited examples.
- Validate data quality, fairness, calibration, and performance on representative current data before considering any real-world use.
