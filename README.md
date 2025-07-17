# Credit Card Fraud Detection using K-Nearest Neighbor (KNN)

This project focuses on detecting fraudulent credit card transactions using a supervised machine learning approach — specifically, the K-Nearest Neighbor (KNN) algorithm. The dataset used contains anonymized transaction features, and the objective is to accurately classify transactions as fraudulent or genuine.

---

## 📁 Project Structure

- `Credit_Card_Fraud_Detection_K_Nearest_Neighbor.ipynb`: Main Jupyter Notebook with code, data processing, visualizations, and model implementation.
- `images/feature_distribution_axes_creditcard_fraud.png`: Visualization of feature distributions across multiple variables (V1–V28, Time, Amount, Class).

---

## 📊 Dataset

- **Description**:
  - **Rows**: 284,807 transactions
  - **Columns**: 31 features (including `Time`, `V1–V28`, `Amount`, and `Class`)
  - **Imbalance**: Only ~0.17% of transactions are fraudulent

---

## 🛠️ Tools & Technologies

- **Language**: Python
- **Libraries**: 
  - `numpy`, `pandas` for data manipulation
  - `matplotlib`, `seaborn` for visualization
  - `sklearn` for modeling and evaluation

---

## 🧠 Machine Learning Algorithm

- **K-Nearest Neighbor (KNN)**:
  - Distance-based classification
  - Sensitive to class imbalance and feature scaling
  - Hyperparameter tuning with K value optimization

---

## 🔍 Workflow Summary

1. **Data Loading & Exploration**:
   - Checked class distribution and feature characteristics
   - Visualized features using subplots

2. **Preprocessing**:
   - Feature scaling using `StandardScaler`
   - Train-test split with stratification

3. **Model Training**:
   - Implemented KNN classifier with optimal K value
   - Evaluated with accuracy, precision, recall, and F1-score

4. **Model Evaluation**:
   - Confusion matrix
   - ROC-AUC score
   - Precision-recall trade-offs

---

## 📈 Results

- The KNN model achieves good performance with careful handling of class imbalance and proper feature scaling.
- **Note**: Since the dataset is highly imbalanced, accuracy alone is not a reliable metric.

---

## ✅ Conclusion

- The KNN approach is a simple yet effective baseline model for credit card fraud detection.
- Additional models (e.g., Logistic Regression, Random Forest, or XGBoost) may be explored for enhanced performance.
- Feature engineering and data balancing techniques like SMOTE can further improve classification outcomes.

---

## 📌 Future Scope

- Implement ensemble models (e.g., Bagging, Boosting)
- Apply dimensionality reduction (e.g., PCA)
- Integrate real-time fraud detection with stream data

---

## 📬 Contact

For any queries or collaboration ideas, feel free to reach out:

**Sudipta Sanyal**  
📧 sudipta.sanyal2004@gmail.com   
📍 Future Institute of Technology

---
