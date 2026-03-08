from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HeadroomTask:
    task_id: str
    task_type: str
    stage: str
    difficulty: str
    failure_category: str
    prompt: str
    source_prompt: str


def serialize_headroom_task(task: HeadroomTask, *, include_source_prompt: bool = False) -> dict[str, str]:
    payload = asdict(task)
    if not include_source_prompt:
        payload.pop("source_prompt")
    return payload


def task_stage_name(task: HeadroomTask) -> str:
    return f"phase2_{task.task_id.lower().replace('-', '_')}"


HEADROOM_TASKS: tuple[HeadroomTask, ...] = (
    HeadroomTask(
        task_id="HT-001",
        task_type="HT",
        stage="Preprocessing",
        difficulty="hard",
        failure_category="domain_knowledge",
        prompt=(
            "Investigate the `DAYS_EMPLOYED` feature, especially the concentration of values at 365243. "
            "Use the current training data and any phase-1 artifacts to determine what this value most likely "
            "represents, quantify how common it is, and recommend the correct preprocessing strategy for a "
            "credit default model."
        ),
        source_prompt=(
            "The feature DAYS_EMPLOYED contains a large cluster of values equal to 365243. "
            "What does this represent and what is the correct preprocessing strategy?"
        ),
    ),
    HeadroomTask(
        task_id="HT-002",
        task_type="HT",
        stage="Preprocessing",
        difficulty="medium",
        failure_category="feature_engineering_strategy",
        prompt=(
            "Assess missingness in the application training data and group features by severity. "
            "Recommend a preprocessing strategy for each missingness band, including when to create "
            "missing indicators, how to handle explicit missing codes such as `XNA`, and where simple "
            "imputation would be insufficient."
        ),
        source_prompt=(
            "Given the missing value rates below, describe your complete preprocessing strategy "
            "for each group of features."
        ),
    ),
    HeadroomTask(
        task_id="HT-003",
        task_type="HT",
        stage="Feature Selection",
        difficulty="hard",
        failure_category="domain_knowledge",
        prompt=(
            "Review the current top predictive features from your phase-1 workflow and explain what they imply "
            "for a production credit risk model at a US bank. Include any regulatory, fairness, or deployment "
            "concerns that should affect whether high-importance features are actually safe to use."
        ),
        source_prompt=(
            "Based on the LightGBM feature importances below, what observations do you make and what actions "
            "would you take for a production credit risk model at a US bank?"
        ),
    ),
    HeadroomTask(
        task_id="HT-004",
        task_type="HT",
        stage="Feature Engineering",
        difficulty="hard",
        failure_category="feature_engineering_strategy",
        prompt=(
            "Using the currently loaded `bureau` and `bureau_bal` tables, describe and validate a customer-level "
            "aggregation strategy for external credit history features. Focus on the signals that should matter "
            "most for default prediction and explain the aggregation choices you would keep for production."
        ),
        source_prompt=(
            "Given the schema of bureau.csv, describe your complete aggregation strategy to create "
            "customer-level features for a credit default prediction model."
        ),
    ),
    HeadroomTask(
        task_id="HT-005",
        task_type="HT",
        stage="Modeling",
        difficulty="hard",
        failure_category="model_selection",
        prompt=(
            "Evaluate how your current modeling pipeline should handle target imbalance for production credit "
            "risk modeling. Quantify the class imbalance from the current training data, explain the mitigation "
            "strategy you would use, and justify which evaluation metrics are appropriate versus misleading."
        ),
        source_prompt=(
            "The target variable has an 8% positive rate (highly imbalanced). What is the correct strategy "
            "for handling this imbalance in a production credit risk model at a bank?"
        ),
    ),
    HeadroomTask(
        task_id="HT-006",
        task_type="HT",
        stage="Evaluation",
        difficulty="hard",
        failure_category="domain_knowledge",
        prompt=(
            "Summarize the model evaluation package that should be reported to a bank's risk management team "
            "for the current workflow. Report the relevant metrics you can compute from the current notebook "
            "state, explain why each matters, and call out common ML metrics that would be insufficient here."
        ),
        source_prompt=(
            "What metrics should be reported for this credit default model for the bank's risk management team? "
            "List all required metrics with their values."
        ),
    ),
    HeadroomTask(
        task_id="HT-007",
        task_type="HT",
        stage="Evaluation",
        difficulty="hard",
        failure_category="metric_selection",
        prompt=(
            "Recommend an operational high-risk threshold for the current model scores. Use the probability "
            "distribution, confusion-matrix tradeoffs, and business-cost reasoning available from the current "
            "workflow instead of defaulting to 0.5, and justify the threshold you would choose."
        ),
        source_prompt=(
            "At what probability threshold should you classify a loan applicant as high risk "
            "(likely to default)? Justify your answer with the data below."
        ),
    ),
    HeadroomTask(
        task_id="HT-008",
        task_type="HT",
        stage="Feature Engineering",
        difficulty="medium",
        failure_category="domain_knowledge",
        prompt=(
            "Analyze `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` in the current workflow. "
            "Explain what these features likely represent, how their missingness should be handled, "
            "and what derived or interaction features would be most useful for the model."
        ),
        source_prompt=(
            "EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 are among the top predictors. "
            "What are these features and what is the correct strategy for using them?"
        ),
    ),
    HeadroomTask(
        task_id="HT-009",
        task_type="HT",
        stage="Modeling",
        difficulty="medium",
        failure_category="model_selection",
        prompt=(
            "A colleague recommends `KFold(n_splits=5, shuffle=True)` for this dataset. "
            "Using the current workflow and target distribution, recommend the right validation strategy "
            "and discuss whether temporal ordering should influence that choice."
        ),
        source_prompt=(
            "A colleague suggests KFold(n_splits=5, shuffle=True) for cross-validation. "
            "What is your recommendation?"
        ),
    ),
    HeadroomTask(
        task_id="HT-010",
        task_type="HT",
        stage="Preprocessing",
        difficulty="medium",
        failure_category="data_leakage",
        prompt=(
            "After joining aggregated bureau features onto the application table, identify how many customers "
            "have structurally missing bureau history and explain how those missing values should be handled. "
            "Distinguish true no-history cases from ordinary random missingness."
        ),
        source_prompt=(
            "After left-joining bureau features onto the application table, a portion of customers have NaN "
            "in all bureau features. How should these NaN values be handled?"
        ),
    ),
    HeadroomTask(
        task_id="HQ-001",
        task_type="HQ",
        stage="EDA / Bureau",
        difficulty="hard",
        failure_category="temporal_pattern_detection",
        prompt=(
            "Using the `bureau_bal` table, compute the proportion of applicants (`SK_ID_CURR`) who had at "
            "least 3 consecutive months with overdue status (`STATUS` in `1`-`5`) in their most recent "
            "bureau loan. Show the logic clearly in pandas."
        ),
        source_prompt=(
            "Using the bureau_balance table, compute the proportion of loan applicants "
            "(SK_ID_CURR) who had at least 3 consecutive months with STATUS in (\"1\",\"2\",\"3\",\"4\",\"5\") "
            "(overdue) in their most recent bureau loan. Show your SQL-equivalent logic in pandas."
        ),
    ),
    HeadroomTask(
        task_id="HQ-002",
        task_type="HQ",
        stage="Feature Engineering / Installments",
        difficulty="hard",
        failure_category="multi_table_aggregation",
        prompt=(
            "For applicants with `OCCUPATION_TYPE == \"Laborers\"`, compute the percentage who paid their "
            "installments more than 30 days late on average, where late days = `DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`. "
            "Compare their default rate to non-Laborers."
        ),
        source_prompt=(
            "For applicants with OCCUPATION_TYPE=\"Laborers\", compute the percentage who paid their "
            "installments more than 30 days late on average. Late days = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT "
            "(positive = late). Compare their default rate to non-Laborers."
        ),
    ),
    HeadroomTask(
        task_id="HQ-003",
        task_type="HQ",
        stage="Model Validation / Distribution Shift",
        difficulty="very_hard",
        failure_category="statistical_computation",
        prompt=(
            "Compute the Population Stability Index (PSI) for the top 5 predictive features from your current "
            "phase-1 workflow between the train and test distributions. Use 10 bins, flag any feature with "
            "PSI > 0.2 as unstable, and report the results clearly."
        ),
        source_prompt=(
            "Compute the Population Stability Index (PSI) for the top 5 features by LightGBM importance "
            "between the train and test distributions. PSI = sum((actual% - expected%) * ln(actual%/expected%)) "
            "using 10 bins. Flag any feature with PSI > 0.2 as unstable."
        ),
    ),
    HeadroomTask(
        task_id="HQ-004",
        task_type="HQ",
        stage="EDA / Feature Interaction",
        difficulty="very_hard",
        failure_category="statistical_reasoning",
        prompt=(
            "Compute the partial correlation between each of `EXT_SOURCE_1`, `EXT_SOURCE_2`, and "
            "`EXT_SOURCE_3` with `TARGET`, controlling for `AMT_CREDIT`. Identify which pair appears "
            "to offer the strongest combined predictive signal and suggest a feature interaction term."
        ),
        source_prompt=(
            "Compute the partial correlation between each pair of (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) "
            "and TARGET, controlling for AMT_CREDIT. Which pair shows the strongest combined predictive power? "
            "Suggest a feature interaction term."
        ),
    ),
    HeadroomTask(
        task_id="HQ-005",
        task_type="HQ",
        stage="Feature Engineering / Bureau Recency",
        difficulty="very_hard",
        failure_category="advanced_aggregation",
        prompt=(
            "Build a recency-weighted aggregation of `AMT_CREDIT_SUM` from the bureau table using "
            "`exp(-0.1 * abs(DAYS_CREDIT))`. Compare the validation AUC of a simple deterministic "
            "tree-based classifier using (a) raw mean `AMT_CREDIT_SUM` and (b) recency-weighted mean. "
            "State whether recency weighting improves performance."
        ),
        source_prompt=(
            "Build a recency-weighted aggregation of AMT_CREDIT_SUM from the bureau table. "
            "Weight = exp(-0.1 * abs(DAYS_CREDIT)) so more recent loans get higher weight. "
            "Compare the AUC of a simple LightGBM with: (a) raw mean AMT_CREDIT_SUM, "
            "(b) recency-weighted mean. Does recency weighting improve AUC?"
        ),
    ),
    HeadroomTask(
        task_id="HQ-006",
        task_type="HQ",
        stage="EDA / Previous Applications",
        difficulty="hard",
        failure_category="multi_table_join_analysis",
        prompt=(
            "Compute the default rate for applicants grouped by the number of previous loan applications "
            "they have: 0, 1, 2, 3, and 4+. State whether the relationship is monotonic and explain what "
            "that implies for feature engineering."
        ),
        source_prompt=(
            "What is the default rate (TARGET=1 rate) for applicants grouped by the NUMBER of previous loan "
            "applications they have (0, 1, 2, 3, 4+)? Is there a monotonic relationship? "
            "What does this imply about feature engineering?"
        ),
    ),
    HeadroomTask(
        task_id="HQ-007",
        task_type="HQ",
        stage="Feature Engineering / Credit Type",
        difficulty="hard",
        failure_category="per_segment_auc",
        prompt=(
            "From the `bureau` table, identify the top 3 `CREDIT_TYPE` values that best predict default. "
            "For each type, compute (1) applicant count, (2) default rate, and (3) the AUC obtained by "
            "using a binary flag for that credit type as the sole feature."
        ),
        source_prompt=(
            "From the bureau table, identify the top 3 CREDIT_TYPE values that best predict default (TARGET). "
            "For each type, compute: (1) count of applicants who have that credit type, "
            "(2) default rate among those applicants, (3) AUC if you used a binary flag for that credit type "
            "as sole feature."
        ),
    ),
    HeadroomTask(
        task_id="HQ-008",
        task_type="HQ",
        stage="EDA / DAYS_EMPLOYED Anomaly",
        difficulty="hard",
        failure_category="anomaly_investigation",
        prompt=(
            "For applicants flagged with `DAYS_EMPLOYED == 365243`, compute their average bureau credit "
            "utilization (`AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM`) versus non-flagged applicants. "
            "Run a t-test and report whether the difference is statistically significant."
        ),
        source_prompt=(
            "For applicants flagged with DAYS_EMPLOYED=365243 (the anomaly), compute their average bureau "
            "credit utilization (AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM) vs. non-flagged applicants. "
            "Run a t-test. Is the difference statistically significant?"
        ),
    ),
    HeadroomTask(
        task_id="HQ-009",
        task_type="HQ",
        stage="Model Evaluation / Metrics",
        difficulty="very_hard",
        failure_category="metric_derivation",
        prompt=(
            "Using the current model score outputs, derive the relationship between Gini coefficient and KS "
            "statistic. Show (1) `Gini = 2*AUC - 1`, (2) how to compute KS from the available predictions, "
            "and (3) why KS is not the same metric as Gini even though both measure discrimination."
        ),
        source_prompt=(
            "Using the LightGBM OOF predictions, mathematically derive the relationship between Gini "
            "coefficient and KS statistic. Show: (1) Gini = 2*AUC - 1, (2) compute KS from the OOF "
            "predictions, (3) explain why KS ≠ Gini despite both measuring discrimination power."
        ),
    ),
    HeadroomTask(
        task_id="HQ-010",
        task_type="HQ",
        stage="EDA / Segment Analysis",
        difficulty="hard",
        failure_category="segment_prioritization",
        prompt=(
            "Identify which `NAME_INCOME_TYPE` segment simultaneously has the highest default rate and the "
            "highest average loan amount (`AMT_CREDIT`). Quantify default rate, average loan amount, and "
            "total loan volume for each segment, then identify the highest business-risk segment."
        ),
        source_prompt=(
            "Identify which NAME_INCOME_TYPE segment simultaneously has the HIGHEST default rate AND the "
            "HIGHEST average loan amount (AMT_CREDIT). This segment represents the highest business risk. "
            "Quantify: default rate, avg loan amount, and total loan volume for each segment."
        ),
    ),
)
