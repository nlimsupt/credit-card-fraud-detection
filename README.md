## credit-card-fraud-detection

### Objective
Detect fraudulent credit card transactions using machine learning models, with a focus on handling extreme class imbalance and minimizing false alarms in real-world financial systems.

### Dataset
- Source: Kaggle – Credit Card Fraud Detection Dataset
- Access: Loaded directly via kagglehub for reproducibility
- Target: Fraudulent transaction (fraud / legitimate)
- Challenge: Extremely imbalanced outcome (fraud ≈ 0.172%)

### Methods
- Logistic Regression
- Random Forest
- XGBoost (cost-sensitive learning)
- SMOTE for class imbalance (Logistic Regression, Random Forest)
- scale_pos_weight for cost-sensitive learning (XGBoost)
- Stratified 5-fold cross-validation

### Evaluation Metrics
- PR-AUC (primary metric)
- Precision (class 1)
- Recall (class 1)
- F1-Score (class 1)

### Tools
Python, scikit-learn, imbalanced-learn, XGBoost, pandas, numpy, matplotlib, seaborn, Google Colab, and Jupyter Notebook

### Key Result
- Logistic Regression achieved high recall but generated a large number of false positives.
- Random Forest improved the balance between precision and recall, outperforming Logistic Regression across imbalance-aware metrics.
- XGBoost with cost-sensitive learning (scale_pos_weight) achieved the best overall performance, delivering the highest PR-AUC and the most favorable precision–recall trade-off.
- Confusion matrix analysis confirmed that XGBoost significantly reduced false alarms while maintaining strong fraud detection capability, making it the most suitable model for real-world deployment.
