# 🧠 Anomaly Detection in Network Traffic (KDD Cup 1999)

This project implements unsupervised anomaly detection techniques on network traffic data using the **KDD Cup 1999** (10% corrected) dataset. The two models explored are:

- 🌲 **Isolation Forest**
- ⚙️ **Deep Autoencoder**

---

## 📂 Project Structure

├── anomaly_detection.ipynb           # Main Jupyter notebook (full pipeline)  
├── kddcup.data_10_percent_corrected  # Dataset file (not uploaded due to size)  
├── autoencoder_model.h5              # Trained Autoencoder model  
├── scaler.pkl                        # StandardScaler object used in preprocessing  
├── reconstruction_error.npy          # Saved reconstruction error for all samples  
├── true_labels.npy                   # Ground truth anomaly labels  
├── best_threshold.npy                # Best threshold determined from PR curve  
├── README.md                         # This file  

---

## 📊 Dataset

- **Source:** KDD Cup 1999 (10% corrected sample)  
- **File used:** `kddcup.data_10_percent_corrected`

---

## ⚙️ Features of This Project

1. **Preprocessing**
   - Label encoding of categorical features
   - Standardization using `StandardScaler`
   - Dimensionality reduction with PCA (for visualization)

2. **Exploratory Data Analysis**
   - Class distribution
   - Top attack types
   - Correlation heatmap
   - Feature-wise boxplots

3. **Anomaly Detection Models**
   - **Isolation Forest:** Outlier detection based on ensemble of trees
   - **Autoencoder:** Neural reconstruction error-based detection

4. **Model Evaluation**
   - Precision, Recall, F1 Score
   - PR Curve analysis and threshold optimization
   - Confusion Matrix heatmap

5. **Model Saving**
   - Autoencoder model, scaler, and detection thresholds saved for reuse

---

## 🧪 Requirements

Install all required packages using:

pip install -r requirements.txt


---

## 🚀 How to Run

1. Download and place `kddcup.data_10_percent_corrected` in the project root.
2. Run the `anomaly_detection.ipynb` notebook step-by-step.
3. Visualize results and performance metrics.
4. Trained models and data are saved automatically at the end of the notebook.

---

## 📈 Sample Output

- Precision-Recall-F1 Threshold Curve
- 2D PCA Scatterplots for both models
- Evaluation reports (Classification + Confusion Matrix)

---

## 📌 License

This project is open-source and free to use for research and educational purposes.
