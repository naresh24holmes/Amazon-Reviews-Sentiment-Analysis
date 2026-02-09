# 📝 Sentiment Analysis on Product Reviews

A machine learning project that classifies product reviews into **Positive** or **Negative** sentiments using **TF‑IDF features**, **Logistic Regression**, and **SMOTE** for class balancing.  
Includes a **Streamlit app** for interactive local deployment.



## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd sentiment-analysis

2. **Install Dependencies**

    pip install -r requirements.txt

3. **Train the Model**  

    python src/train.py 

4. **Run the Streamlit App** 

    streamlit run src/app.py 

**Usage** 

Open the app in your browser (default: http://localhost:8501).

Enter a product review in the text box.

Click Predict to see whether the sentiment is Positive 😊 or Negative 😡.

