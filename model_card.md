---
license: MIT
language:
- en
datasets:
- iris
metrics:
- accuracy
- f1
pipeline_tag: tabular-classification
---

## Model Details
- Model name: Iris classifier
- Author: Jabiko
- Date: October 18, 2026
- version: 1.0
- Model type: RandomForestClassifier (scikit-learn)

## Intended Use

### Intended Use Cases
- **Purpose:** Classify an iris flower into one of three species (*setosa*, *versicolor*, *virginica*) from four numeric measurements in centimeters: sepal length, sepal width, petal length and petal width.
- **Intended users:** Students, teachers and ML practitioners learning about supervised classification, model documentation (Model Cards) and AI governance practices such as AI Act transparency and traceability requirements.
- **Context:** Educational and demonstration use only, within the AI Act course (chapter 2). It is a reference example for training, evaluating, documenting and publishing a model on the Hugging Face Hub.

### Out-of-scope Use
- **Production use:** The model is not designed, tested or monitored for production deployment. There is no versioned serving pipeline, no drift monitoring and no input validation.
- **High-stakes decisions:** It must not be used for any decision with real consequences for people, organizations or the environment, including the high-risk use cases in Annex III of the AI Act.
- **Real botanical identification:** It is not suitable for scientific, agricultural or commercial species identification. It was trained on 120 samples from a single historical dataset (Fisher, 1936) and has not been validated on field data.
- **Other species or inputs:** It cannot recognize iris species other than the three classes above, or any other plant. It always returns one of the three classes, even for inputs that are out of distribution or invalid, such as non-iris measurements, other units or negative values.
- **Other data types:** It is not designed for images, text or any input other than the four tabular features listed above.

## Factors (Facteurs)
Two main factors can affect the model's performance. The first is the **distribution of the input features**. Petal length and width separate *setosa* clearly from the other two species (petal length 1.0–1.9 cm vs. 3.0–6.9 cm). However, *versicolor* and *virginica* overlap on every feature, for example petal length 3.0–5.1 cm vs. 4.5–6.9 cm. Most errors are therefore expected in that overlap zone. Performance may also drop for measurements outside the ranges seen during training (e.g. sepal length 4.3–7.7 cm, petal length 1.0–6.7 cm). This can happen with flowers grown in other conditions, or with different measurement tools or protocols. The second factor is **class balance**. The Iris dataset is perfectly balanced (50 samples per species). The split is not stratified, however, so the training set has 40 / 41 / 39 samples and the test set 10 / 9 / 11 (setosa / versicolor / virginica). With so few test samples per class, a single misclassification changes per-class metrics by about 10 points. Results may also differ on real-world data where species frequencies are not equal.

## Metrics (Métriques)
Output of `train.py` on the test set (30 samples):

```text
Accuracy : 1.0000

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

**Why these metrics:**
- **Accuracy** gives the overall share of correct predictions. It is easy to interpret and meaningful here because the classes are roughly balanced.
- **Precision and recall per class** show how each species is handled. A multi-class model can score well overall and still confuse two specific classes, such as *versicolor* and *virginica*. Precision measures how reliable a prediction is, and recall measures how many examples of a class are found.
- **F1-score** is the harmonic mean of precision and recall. It summarizes both in a single number per class.
- **Macro average** weights every class equally, so a weak class cannot hide behind the others. **Weighted average** takes class support into account, which matters when the test set is not perfectly balanced (10 / 9 / 11).

A perfect score on only 30 test samples should not be read as a guarantee of perfect performance in general (see Factors and Caveats).

## Training Data (Données d'entraînement)
- **Dataset name:** Iris flower dataset (Fisher, 1936; measurements collected by Edgar Anderson)
- **Source:** `sklearn.datasets.load_iris` (scikit-learn's bundled copy of the UCI Machine Learning Repository dataset)
- **License:** CC BY 4.0 (UCI Machine Learning Repository)
- **Full dataset size:** 150 samples (50 per species), no missing values
- **Training set size:** 120 samples (80 % of the dataset), produced by `train_test_split(test_size=0.2, random_state=42)`
- **Class distribution in the training set:** 40 setosa / 41 versicolor / 39 virginica (split not stratified)
- **Features (inputs):** 4 continuous numeric variables, in centimeters (min to max observed in the training set):
  - `sepal length (cm)`: 4.3 to 7.7
  - `sepal width (cm)`: 2.0 to 4.4
  - `petal length (cm)`: 1.0 to 6.7
  - `petal width (cm)`: 0.1 to 2.5
- **Target:** iris species, 3 classes encoded as integers: `0` = setosa, `1` = versicolor, `2` = virginica
- **Preprocessing:** None. Raw features are passed directly to the model, with no scaling, feature engineering or data augmentation (Random Forests do not need feature scaling).
- **Personal data:** None. The dataset contains only botanical measurements.

## Evaluation Data (Données d'évaluation)
- **Source:** Same dataset as training (Iris flower dataset, `sklearn.datasets.load_iris`), test split
- **Size:** 30 samples (20 % of the dataset)
- **Class distribution:** 10 setosa / 9 versicolor / 11 virginica
- **Split method:** Random train/test split, 80 % / 20 %, using `train_test_split(X, y, test_size=0.2, random_state=42)`. The split is not stratified, and `random_state=42` makes it reproducible. The test samples are never seen during training.
- **Preprocessing:** None. The same raw features as in training are used, with no specific preprocessing.
- **Limitation:** The test set comes from the same distribution as the training data and is very small. It does not measure how the model generalizes to iris measurements collected in other conditions (see Factors).

## Quantitative Analyses (Analyses quantitatives)
Per-class performance on the test set (from the scikit-learn classification report printed by `train.py`):

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| setosa     | 1.00      | 1.00   | 1.00     | 10      |
| versicolor | 1.00      | 1.00   | 1.00     | 9       |
| virginica  | 1.00      | 1.00   | 1.00     | 11      |
| **Macro avg** | 1.00   | 1.00   | 1.00     | 30      |

Confusion matrix (rows = true class, columns = predicted class):

| True \ Predicted | setosa | versicolor | virginica |
|------------------|--------|------------|-----------|
| setosa           | 10     | 0          | 0         |
| versicolor       | 0      | 9          | 0         |
| virginica        | 0      | 0          | 11        |

**Interpretation:** Performance is homogeneous across the three classes. Every test sample is classified correctly, and no class is classified worse than the others. On this split, the model does not confuse *versicolor* and *virginica*, even though their features overlap (see Factors). With only 9 to 11 test samples per class, however, these per-class results are statistically fragile, because a single error would lower the class recall by about 10 points. The model has not been evaluated on other subgroups, such as other collection sites or measurement protocols, because the dataset provides no such metadata.

## Ethical Considerations (Considérations éthiques)
- **Usage risks:** The model processes no personal data and makes no decisions about people, so its direct ethical risk is low. The main risk is misuse: the model could be reused outside its educational scope because of its perfect test scores. Examples include automated plant identification for agriculture, conservation work or the detection of protected or toxic species. It is not suited to these uses. It is not validated on real-world data, and it always predicts one of the three classes, even for an unknown species or an invalid measurement. It also gives no reliable signal of uncertainty. Any use beyond education would require new data, a proper validation and human oversight.
- **Potential biases:** The dataset is not representative of iris flowers in general.
  - **Size and geography:** It contains only 150 flowers (50 per species). According to Fisher's paper, the setosa and versicolor samples were collected in the Gaspé Peninsula (Québec, Canada) by a single person with the same instrument.
  - **Under-represented groups:** Other iris species, other regions, other climates and growing conditions, and measurements taken by other people or with other tools are absent. The model may therefore perform worse on them.
  - **Class balance:** The perfect balance between classes (50 / 50 / 50) is artificial and does not reflect how often each species actually occurs.
- **Historical context:** The dataset was published by R. A. Fisher in 1936 in the *Annals of Eugenics*. This does not affect the model's predictions, but it is worth knowing when documenting the data provenance.

## Caveats and Recommendations (Limites et recommandations)
- **Very small dataset (150 samples, 30 for testing).** The perfect scores (1.00 accuracy) come from only 30 test samples. They do not prove that the model will perform perfectly in general.
  → **Recommendation:** Do not generalize these results without additional validation on a larger, independent dataset.
- **Data not representative of the real world.** The samples come from a single historical collection, with a single collector, region and measurement protocol, and cover only three iris species.
  → **Recommendation:** Test the model on data collected in varied conditions (other regions, seasons, collectors and instruments) before any use beyond education.
- **No cross-validation.** The model is evaluated on a single train/test split (`random_state=42`, not stratified). Another split could give different results.
  → **Recommendation:** Use a more robust evaluation before any deployment, such as stratified k-fold cross-validation with confidence intervals on the metrics.
- **No handling of out-of-distribution inputs.** The model always predicts one of the three classes, even for another species or for invalid or out-of-range measurements.
  → **Recommendation:** Add input validation (expected ranges, units) and a rejection mechanism, for example a threshold on predicted probabilities. Keep a human in the loop for any decision.

## Reproductibility
- **git_sha:** `1c9a7c272a24c8a5900868b23960534735e3a2dd`
