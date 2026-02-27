# Healthcare Anomaly Detection System using Autoencoder and Flask

## Project Overview

This project presents a **Healthcare Anomaly Detection System** that analyzes patient health records and identifies abnormal patterns using a **deep learning Autoencoder model**.

The system is integrated with a **Flask web application** that allows users to upload patient data in **CSV format**. The trained model processes the data, detects anomalies in health parameters, and generates **medical suggestions based on abnormal values**.

The results are automatically saved and provided as a **downloadable report**, making it easier for healthcare professionals to review patient health risks.

---

# Introduction

Monitoring patient health parameters is essential for **early detection of medical risks**. Traditional manual analysis of health data can be time-consuming and may overlook subtle anomalies.

This project uses an **Autoencoder-based anomaly detection model** to identify irregular health patterns from patient datasets. The system evaluates parameters such as blood pressure, glucose level, heart rate, cholesterol, and age to determine potential health concerns.

The solution provides **automated suggestions based on detected anomalies**, helping healthcare professionals make informed decisions.

---

# Key Features

• Upload patient health data through a web interface
• Deep learning–based anomaly detection
• Automatic identification of abnormal health records
• Suggestion generation based on medical thresholds
• Downloadable results in CSV format
• Flask-based interactive web application

---

# Health Parameters Used

The system analyzes the following patient health metrics:

| Parameter    | Description              |
| ------------ | ------------------------ |
| BP_Systolic  | Systolic Blood Pressure  |
| BP_Diastolic | Diastolic Blood Pressure |
| Glucose      | Blood glucose level      |
| Heart_Rate   | Patient heart rate       |
| Cholesterol  | Blood cholesterol level  |
| Age          | Patient age              |

---

# Threshold Values

The model generates suggestions based on predefined medical thresholds.

| Parameter    | Threshold |
| ------------ | --------- |
| BP_Systolic  | 130       |
| BP_Diastolic | 85        |
| Glucose      | 140       |
| Heart_Rate   | 100       |
| Cholesterol  | 200       |
| Age          | 65        |

If a patient's value exceeds these limits, the system generates **medical recommendations**.

---

# Model Architecture

The system uses a **Deep Learning Autoencoder model**.

### What is an Autoencoder?

An autoencoder is an **unsupervised neural network used for anomaly detection**. It learns the normal patterns in the dataset and attempts to reconstruct the input data.

If the reconstruction error is high, the sample is considered **anomalous**.

### Working Principle

1. The model learns patterns from normal patient data.
2. When new data is uploaded, the model reconstructs the input.
3. The **reconstruction error** is calculated.
4. If the error exceeds a threshold, the record is flagged as an anomaly.

---

# System Workflow

1. User uploads a **CSV file containing patient health records**.
2. The Flask application loads the trained **Autoencoder model**.
3. The data is processed and passed through the model.
4. Reconstruction error is calculated for each sample.
5. If abnormal values are detected, the system generates **health suggestions**.
6. Results are saved as a **CSV file**.
7. The user can **download the generated report**.

---

# Example Suggestions Generated

If anomalies are detected, the system may generate suggestions such as:

• Monitor blood pressure regularly and consult a cardiologist
• Review diet and consider diabetes screening
• Check for stress or arrhythmia symptoms
• Implement a heart-healthy diet
• Schedule regular health check-ups due to age

---

# Project Structure

```
Healthcare-Anomaly-Detection/
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   └── index.html
│
├── autoencoder_model.py
├── autoencoder_anomaly_detection.pth
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation

### Clone the Repository

```
git clone https://github.com/yourusername/healthcare-anomaly-detection.git
```

### Navigate to the Project Folder

```
cd healthcare-anomaly-detection
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

# Running the Application

Start the Flask server:

```
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000/
```

Upload the CSV file to analyze patient health records.

---

# Input File Format

The uploaded CSV file should contain the following columns:

```
BP_Systolic
BP_Diastolic
Glucose
Heart_Rate
Cholesterol
Age
```

Example:

```
BP_Systolic,BP_Diastolic,Glucose,Heart_Rate,Cholesterol,Age
120,80,110,72,180,45
145,90,160,105,230,70
```

---

# Output

The system generates a downloadable **CSV report containing medical suggestions**.

Example Output:

| Sample ID | Suggestions                                                                   |
| --------- | ----------------------------------------------------------------------------- |
| 0         | No anomaly detected                                                           |
| 1         | Monitor blood pressure regularly; Review diet and consider diabetes screening |

---

# Technologies Used

• Python
• Tensorflow
• Flask
• NumPy
• Pandas

---

# Applications

• Patient health monitoring
• Early detection of medical risks
• Hospital data analysis
• Healthcare decision support systems

---

# Future Improvements

• Real-time patient monitoring system
• Integration with hospital electronic health records
• Visualization dashboards for anomaly trends
• Deployment on cloud platforms

---

# Author

**Tharun Kumar**
