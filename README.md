# GNN-Based Real-Time Fraud Detection Web Application

![<img width="1439" height="859" alt="Screenshot 2025-09-02 141914" src="https://github.com/user-attachments/assets/49533bcc-5271-40f0-9f86-2d7828d72282" />
) 
This project is an end-to-end application that uses a Graph Neural Network (GNN) to detect fraudulent credit card transactions from a real-world imbalanced dataset. The trained model is served via a Python backend and can be tested through a professional, interactive web interface.

## Key Features

-   **High-Performance GNN Model:** Utilizes a **GraphSAGE** model trained on the full dataset of ~284,000 transactions. It proved highly effective, achieving a **Test F1-Score of 0.82**.
-   **Simulated Live Mode:** A user-friendly interface that accepts simple inputs (`Time`, `Amount`) and uses a secondary `RandomForest` model to realistically predict the complex, anonymized features required by the GNN for a live prediction.
-   **Guided Demo Mode:** One-click buttons to simulate real legitimate and fraudulent transactions, allowing anyone to easily test and verify the model's high-confidence predictions.
-   **FastAPI Backend:** A robust and modern API built with FastAPI to serve the machine learning models. Implements "lazy loading" to manage memory usage efficiently on consumer hardware.
-   **Modern Frontend:** A professional and responsive user interface built with HTML, CSS, and JavaScript, inspired by modern AI application designs.

## Technologies Used

-   **Backend:** Python, FastAPI, Uvicorn
-   **Machine Learning:** PyTorch, PyTorch Geometric, Scikit-learn, Pandas, NumPy
-   **Frontend:** HTML, CSS, JavaScript

## Local Setup and Installation

To run this project locally, please follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/Ankurmishra05/gnn-fraud-detection-app.git](https://github.com/Ankurmishra05/gnn-fraud-detection-app.git)
cd gnn-fraud-detection-app
```

**2. Create and activate a virtual environment:**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate  
```

**3. Install dependencies:**
This project requires specific library versions to ensure model compatibility. Please install them from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Generate Model Artifacts:**
The trained models are not stored in this repository due to their size. You must first run the included Jupyter notebook (`fraud_detection_notebook.ipynb`) to train the models and generate the necessary files. This notebook will create all the required `.pth` and `.pkl` files.

## How to Run the Web Application

1.  **Start the Backend Server:**
    Once the model artifacts are generated and placed in the main folder, run the following command in your terminal:
    ```bash
    uvicorn main:app --reload
    ```
    The server will be running at `http://127.0.0.1:8000`.

2.  **Launch the Frontend:**
    - If you are using VS Code, install the "Live Server" extension by Ritwick Dey.
    - Right-click on the `MishraAnkur_GNN_Fraud_API.html` file and select "Open with Live Server".
    - Your browser will open to the application's user interface.
