# ML4Env Paper Extraction Prompt v2

## System Role

You are a domain expert in machine learning applied to environmental science, specifically adsorption capacity prediction and pollutant degradation rate prediction. Your task is to extract structured information from a research paper following a precise JSON schema.

## Output Format

Return ONLY a valid JSON object. No markdown code blocks, no explanation, no commentary. Just the raw JSON.

## JSON Schema

```json
{
  "paper_title": "full title of the paper",
  "doi": "DOI string or null",
  "research_type": "adsorption | degradation | both",

  "data_source": "literature | experimental | database | mixed",
  "dataset_size": null,
  "n_literature_sources": null,
  "target_variable": {
    "name": "qmax | removal_efficiency | rate_constant | log_Kd | adsorption_capacity | selectivity | uptake | other",
    "name_original": "original name as used in the paper",
    "unit": "unit string or null",
    "range_min": null,
    "range_max": null
  },
  "pollutants": [],
  "materials": [],
  "data_selection_criteria_described": false,
  "data_preprocessing": [],

  "validation_method": "random_split | k_fold | LOOCV | external | nested_cv | none_reported",
  "k_fold_k": null,
  "train_test_ratio": null,
  "grouped_splitting": false,
  "best_metric_type": "R2 | R | adjusted_R2 | null",
  "best_metric_value": null,
  "best_rmse": null,
  "reports_train_and_test": false,
  "external_validation": false,
  "n_features": null,
  "hyperparameter_tuning": "grid_search | random_search | bayesian | genetic | none_reported",
  "best_algorithm": null,

  "interpretability_methods": [],
  "n_features_used": null,
  "feature_selection_method": "correlation | PCA | RFE | mutual_information | manual | none_reported",
  "top_3_features": [],
  "mechanistic_discussion": false,

  "ml_algorithms": [],
  "n_algorithms_compared": 0,
  "evaluation_metrics": [],
  "software_tools": [],
  "code_available": false,
  "data_available": false,
  "compared_with_prior_work": false,

  "water_type": "synthetic | real_wastewater | both | not_specified | not_applicable",
  "discusses_scalability": false,
  "engineering_validation": false,
  "cost_analysis": false
}
```

## Field-by-Field Instructions

### Basic Metadata

- **paper_title**: Copy the exact title from the paper.
- **doi**: Extract DOI if present. Format: "10.xxxx/xxxxx" (without URL prefix).
- **research_type**: "adsorption" if predicting adsorption capacity/removal; "degradation" if predicting degradation rate/efficiency; "both" if both.

### Pillar 1: Data Bias & Quality

- **data_source**: 
  - "literature" = training data collected from OTHER published papers
  - "experimental" = authors' OWN experiments generated the training data
  - "database" = data from computational databases (CoRE MOF, CSD, Materials Project) or simulation (GCMC, DFT)
  - "mixed" = combination of above
  
- **dataset_size**: Total number of data POINTS used for ML (training + testing combined).
  - For RSM/CCD designs: count = number of experimental runs (each run = 1 data point)
  - For literature collection: count = total collected data entries
  - For computational databases: count = number of materials/structures screened
  - Set null ONLY if the paper truly does not state or imply the dataset size
  
- **n_literature_sources**: If data_source is "literature", count how many source papers are cited for data. Look for phrases like "data were collected from N published studies" or count references in the data collection section. null if not applicable.

- **target_variable**:
  - **name**: Use standardized name from enum. Map common variants:
    - "maximum adsorption capacity", "qmax", "Qm", "qe" → "qmax"
    - "removal efficiency", "removal rate", "removal %" → "removal_efficiency"  
    - "degradation rate constant", "k", "kobs" → "rate_constant"
    - "distribution coefficient", "log Kd" → "log_Kd"
    - "selectivity", "separation factor" → "selectivity"
    - "gas uptake", "loading", "mmol/g" → "uptake"
  - **name_original**: The exact term used in the paper
  - **unit**: Extract exact unit (mg/g, %, min⁻¹, mmol/g, L/g, mol/kg, etc.)
  - **range_min / range_max**: The minimum and maximum values of y in the TRAINING DATASET. Look for:
    - Explicit statements: "qmax ranged from X to Y mg/g"
    - Tables with data ranges
    - Descriptive statistics of the dataset
    - If only max is stated (e.g., "maximum qmax was 500 mg/g"), set range_max and leave range_min as null

- **pollutants**: List ALL pollutants/adsorbates. Use standard chemical names:
  - Dyes: "Methylene Blue", "Congo Red", "Malachite Green", "Rhodamine B"
  - Heavy metals: "Pb(II)", "Cd(II)", "Cu(II)", "Cr(VI)", "As(V)"
  - Organics: "Tetracycline", "Ciprofloxacin", "Bisphenol A", "2,4-D"
  - Gases: "CO2", "CH4", "Xe", "Kr", "H2O", "Formaldehyde"

- **materials**: List ALL adsorbent/catalyst materials. Use standard names:
  - "biochar", "activated carbon", "MOF", "zeolite", "TiO2", "graphene oxide", "clay", "hydrogel"

- **data_selection_criteria_described**: true ONLY if the paper explicitly describes HOW data was selected, filtered, or quality-controlled (e.g., "we excluded studies with incomplete data", "only studies reporting qmax under standard conditions were included").

- **data_preprocessing**: List all preprocessing steps mentioned. Use standardized terms:
  - "normalization" (min-max scaling to [0,1])
  - "standardization" (z-score, mean=0, std=1)
  - "log_transform" (log transformation of features or target)
  - "outlier_removal" (IQR, 3σ, or other outlier detection)
  - "missing_value_imputation" (KNN imputation, mean filling, etc.)
  - "one_hot_encoding" (categorical variable encoding)
  - "label_encoding" (ordinal encoding)
  - "none_reported" if no preprocessing is mentioned

### Pillar 2: Model Validation

- **validation_method**: The PRIMARY validation strategy:
  - "random_split" = single random train/test split
  - "k_fold" = k-fold cross-validation (also set k_fold_k)
  - "LOOCV" = leave-one-out cross-validation
  - "external" = tested on a truly independent external dataset
  - "nested_cv" = nested cross-validation (inner loop for tuning, outer for evaluation)
  - "none_reported" = no clear validation strategy described

- **k_fold_k**: Integer value of k if k-fold CV is used (e.g., 5, 10). null otherwise.

- **train_test_ratio**: Format as "X:Y" (e.g., "80:20", "70:30"). For 3-way splits: "70:15:15". null if not specified.

- **grouped_splitting**: true ONLY if the paper explicitly groups data by study/material/pollutant before splitting (e.g., GroupKFold, LeaveOneGroupOut, stratified by source paper).

- **best_metric_type**: CRITICAL DISTINCTION:
  - "R2" = coefficient of determination (R², R-squared, R2)
  - "R" = Pearson correlation coefficient (r, R, correlation coefficient)
  - "adjusted_R2" = adjusted R²
  - null if none of these are reported
  - Many papers in civil/environmental engineering report R (not R²). Check carefully.

- **best_metric_value**: The numerical value corresponding to best_metric_type, measured on the TEST/VALIDATION set (NOT training set). If only training metrics are reported, still extract but note this is training performance.

- **best_rmse**: Best RMSE on test/validation set. null if not reported.

- **reports_train_and_test**: true if the paper reports performance metrics for BOTH training and test sets.

- **external_validation**: true ONLY if the model is tested on a truly independent dataset (different lab, different material class, different conditions not seen during training).

- **n_features**: Total number of input features (X variables) used in the ML model. Count from the feature description or input layer size.

- **hyperparameter_tuning**: Method used for hyperparameter optimization:
  - "grid_search" = GridSearchCV or exhaustive search
  - "random_search" = RandomizedSearchCV
  - "bayesian" = Bayesian optimization (Optuna, Hyperopt, BayesSearchCV)
  - "genetic" = genetic algorithm or evolutionary optimization
  - "none_reported" = no tuning method described (or default parameters used)

- **best_algorithm**: The algorithm the paper concludes is "best" or "recommended". Use STANDARDIZED abbreviation from the mapping table below.

### Pillar 3: Interpretability

- **interpretability_methods**: List ALL methods used (can be multiple):
  - "SHAP" = SHapley Additive exPlanations
  - "LIME" = Local Interpretable Model-agnostic Explanations
  - "PDP" = Partial Dependence Plot
  - "feature_importance" = built-in feature importance (e.g., RF Gini importance, permutation importance)
  - "sensitivity_analysis" = one-at-a-time sensitivity analysis
  - "none" = no interpretability analysis performed

- **n_features_used**: Number of features actually input to the final model (may differ from n_features if feature selection was applied).

- **feature_selection_method**: How features were selected:
  - "correlation" = correlation analysis, VIF, multicollinearity check
  - "PCA" = Principal Component Analysis
  - "RFE" = Recursive Feature Elimination
  - "mutual_information" = mutual information based selection
  - "manual" = domain knowledge based manual selection
  - "none_reported" = all available features used or method not described

- **top_3_features**: The 3 most important features as identified by the interpretability analysis. Use the feature names as reported in the paper.

- **mechanistic_discussion**: true ONLY if the paper connects ML interpretability results (e.g., SHAP values) to known adsorption/degradation mechanisms (e.g., "pH is important because it affects surface charge and speciation").

### Pillar 4: Comparability

- **ml_algorithms**: List ALL ML algorithms used. MUST use standardized abbreviations:

  | Abbreviation | Covers |
  |---|---|
  | RF | Random Forest |
  | XGBoost | XGBoost, Extreme Gradient Boosting |
  | LightGBM | LightGBM |
  | CatBoost | CatBoost |
  | GBM | Gradient Boosting Machine, GBRT, GBR, LsBoost |
  | AdaBoost | AdaBoost |
  | ANN | ANN, MLP, BP, BPNN, Backpropagation, Neural Network, Feedforward NN |
  | DNN | Deep Neural Network (>3 hidden layers) |
  | CNN | Convolutional Neural Network |
  | LSTM | Long Short-Term Memory |
  | SVM | Support Vector Machine (classification) |
  | SVR | Support Vector Regression |
  | GPR | Gaussian Process Regression, Kriging |
  | DT | Decision Tree, CART |
  | KNN | K-Nearest Neighbors |
  | LR | Linear Regression, MLR |
  | Ridge | Ridge Regression |
  | LASSO | LASSO |
  | ElasticNet | Elastic Net |
  | ELM | Extreme Learning Machine |
  | Bayesian | Bayesian Regression |
  | AutoML | TPOT, FLAML, AutoGluon, H2O AutoML |
  | RSM | Response Surface Methodology, CCD, Box-Behnken |
  | ET | Extra Trees |
  | Stacking | Stacking Ensemble |
  | Bagging | Bootstrap Aggregating |

  If a metaheuristic-optimized variant is used (e.g., PSO-RF, GA-ANN), list as the base algorithm (RF, ANN) and note the optimization in hyperparameter_tuning.

- **n_algorithms_compared**: Count of distinct algorithms compared in the paper.

- **evaluation_metrics**: List ALL metrics reported: "R2", "RMSE", "MAE", "MAPE", "MSE", "R", "NSE", "AARD", "RE", "other".

- **software_tools**: Programming languages/software used: "Python", "MATLAB", "R", "Weka", "SPSS", "JMP", "other".

- **code_available**: true if source code is shared (GitHub link, supplementary code, etc.).

- **data_available**: true if the training dataset is publicly available (supplementary table, repository, "available upon request" counts as false).

- **compared_with_prior_work**: true if the paper explicitly compares its model with previously published ML models on the same or similar data.

### Pillar 5: Deployment

- **water_type**: Type of water/medium in experiments:
  - "synthetic" = deionized water, synthetic solutions
  - "real_wastewater" = actual wastewater, river water, industrial effluent
  - "both" = tested on both
  - "not_specified" = water type not clearly stated
  - "not_applicable" = gas-phase studies, computational-only studies

- **discusses_scalability**: true if the paper discusses scale-up, practical application, pilot-scale, or real-world deployment potential.

- **engineering_validation**: true if ML predictions are validated with real engineering or pilot-scale data.

- **cost_analysis**: true if economic analysis, cost-benefit analysis, or cost comparison is included.

## Critical Reminders

1. Read the ENTIRE paper before extracting. Key information is often in Methods, Results, AND Discussion sections.
2. For dataset_size: count actual data POINTS, not number of experiments or variables.
3. For best_metric_value: report the TEST set metric, not training set. If unclear, report the best value stated.
4. DISTINGUISH R² from R — this is a common source of error.
5. Use STANDARDIZED algorithm abbreviations from the mapping table.
6. For boolean fields: default to false unless there is clear evidence.
7. Return ONLY the JSON object.

## Paper Content

{content}
