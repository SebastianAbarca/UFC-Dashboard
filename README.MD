### UFC PPV Prediction & Analytics Dashboard

This project is a comprehensive data analysis and machine learning dashboard built with Streamlit. It explores the factors that influence UFC Pay-Per-View (PPV) buys by analyzing historical event data, fighter popularity, and audience engagement with "UFC Embedded" YouTube videos. The dashboard provides interactive visualizations and a predictive model to forecast PPV buys for upcoming events.

---

### Key Features

* **PPV Prediction Model:** An **XGBoost Regressor** trained to predict PPV buys for events. The model is also used to impute missing PPV data for events where the information is not publicly available.
* **Fighter Performance Analysis:** Interactive charts to explore historical PPV data, including top fighters by total buys, event-by-event comparisons, and career PPV trends for individual fighters.
* **YouTube Engagement Correlation:** Visualizations that explore the relationship between "UFC Embedded" YouTube video views and PPV buys, helping to quantify the impact of pre-fight media hype.
* **Model Comparison:** Analysis comparing the performance of different regression models (**XGBoost vs. Ridge Regression**) to determine the most effective predictive approach.
* **Residual Analysis:** Charts that highlight events where the model over- or under-predicted PPV buys, providing insights into potential unmeasured factors.

---

### Technical Stack

* **Python:** The core language for the project.
* **Streamlit:** Used to create the interactive web dashboard.
* **Pandas:** For data loading, cleaning, and manipulation.
* **Scikit-learn:** For machine learning preprocessing and model evaluation.
* **XGBoost:** The primary regression model for PPV prediction.
* **Plotly:** For creating all interactive data visualizations.

---

### How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-project-repo.git](https://github.com/YourUsername/your-project-repo.git)
    cd your-project-repo
    ```
2.  **Set up the virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    *(Note: You will need a `requirements.txt` file containing all the necessary libraries like `streamlit`, `pandas`, `plotly`, `xgboost`, etc.)*
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

### Data Sources

* **UFC Event Data:** Acquired from a custom data source (e.g., a CSV file, API).
* **YouTube Video Views:** Data on "UFC Embedded" videos.

---

### Contact

For any questions or feedback on this project, please feel free to reach out.

* **Name:** [Your Name]
* **GitHub:** [Link to your GitHub Profile]
* **LinkedIn:** [Link to your LinkedIn Profile]