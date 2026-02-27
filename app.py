from flask import Flask, render_template, request, send_file
import pandas as pd
import torch
import numpy as np
import os
from autoencoder_model import Autoencoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Define thresholds and load model
thresholds = {
    "BP_Systolic": 130,
    "BP_Diastolic": 85,
    "Glucose": 140,
    "Heart_Rate": 100,
    "Cholesterol": 200,
    "Age": 65
}

input_dim = 6  # Assuming 6 features
autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load('autoencoder_anomaly_detection.pth'))
autoencoder.eval()

# Function to detect anomalies and provide suggestions
def detect_anomaly_and_suggest(data, model, threshold=0.05):
    data_tensor = torch.FloatTensor(data)
    with torch.no_grad():
        reconstructed = model(data_tensor)
        reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()
        anomalies = (reconstruction_errors > threshold).astype(int)
        # Convert errors to percentages and round to two decimal places
        reconstruction_errors = np.round(reconstruction_errors * 100, 2)
    return anomalies, reconstruction_errors

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load data and process
        data = pd.read_csv(file_path)
        if 'Anomaly_Flag' in data.columns:
            data = data.drop(columns=['Anomaly_Flag'])

        # Assume data is already scaled
        data_scaled = data.values  # Replace this with actual scaling if needed
        anomalies, errors = detect_anomaly_and_suggest(data_scaled, autoencoder)

        # Prepare CSV results (No error display)
        csv_results = []
        # Iterate through results to print suggestions based on each detected anomaly
        for i, is_anomaly in enumerate(anomalies):
            suggestions = []
            if is_anomaly:
                print(f"Sample {i} detected as anomaly with error {errors[i]:.4f}. Suggested actions:")
                
                # Check feature values and print relevant suggestions
                if data.iloc[i]["BP_Systolic"] > thresholds["BP_Systolic"] or data.iloc[i]["BP_Diastolic"] > thresholds["BP_Diastolic"]:
                    suggestions.append("  - Suggestion: Monitor blood pressure regularly and consult a cardiologist.")
                
                if data.iloc[i]["Glucose"] > thresholds["Glucose"]:
                    suggestions.append(" - Suggestion: Review diet and consider a diabetes screening.")
                
                if data.iloc[i]["Heart_Rate"] > thresholds["Heart_Rate"]:
                    suggestions.append("  - Suggestion: Check for signs of stress or arrhythmia, and consult a doctor if symptoms persist.")
                
                if data.iloc[i]["Cholesterol"] > thresholds["Cholesterol"]:
                    suggestions.append("  - Suggestion: Implement a heart-healthy diet and get a lipid profile test.")
                
                if data.iloc[i]["Age"] > thresholds["Age"]:
                    suggestions.append("  - Suggestion: Regular check-ups recommended due to age.")

                # Combine unique suggestions only if they were triggered by the data
                suggestion_text = "; ".join(suggestions) if suggestions else "No specific issues detected"
            else:
                suggestion_text = "No anomaly detected"

            # Add data for CSV output (No error, no anomaly flag)
            csv_results.append({
                "sample_id": i,
                "suggestions": suggestion_text
            })
        
        # Convert CSV results to DataFrame
        results_df = pd.DataFrame(csv_results)
        
        # Save results as a CSV file
        result_filename = f"anomaly_detection_results.csv"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        results_df.to_csv(result_path, index=False)

        # Provide file download link
        return render_template("index.html", download_link=result_filename)
    
    return render_template("index.html", download_link=None)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    # Ensure result folder exists
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
