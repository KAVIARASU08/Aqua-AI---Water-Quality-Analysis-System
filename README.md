<h1 align="center">💧 AI-Powered Water Quality Analysis & Disease Prediction System</h1>

<p align="center">
  An intelligent Flask-based web application that analyzes water quality parameters using Deep Learning models to predict water potability and potential water-borne diseases, while maintaining historical reports and interactive analytics.
</p>

---

## 📌 Overview

Water contamination is one of the leading causes of health issues worldwide. This project uses Artificial Intelligence and Deep Learning to analyze laboratory water sample parameters and determine:

- ✅ Whether the water is safe for drinking
- 🦠 Possible diseases caused by contaminated water
- 📊 Statistical insights and historical analysis
- 📄 Downloadable water quality reports
- 🤖 AI-powered assistant for user guidance

The application provides an intuitive dashboard for managing water sample analysis with interactive visualizations and report generation.

---

## 🚀 Features

### 💧 Water Potability Prediction
- Predicts whether water is safe for drinking
- Uses Deep Learning neural network model
- Provides confidence score for predictions

### 🦠 Disease Prediction
- Predicts possible water-borne diseases
- Multi-class Deep Learning classification
- Risk level identification

### 📊 Analytics Dashboard
- Historical prediction records
- Statistical summaries
- Visual comparison of water samples
- Session management

### 📄 Report Generation
- Generate downloadable water quality reports
- View detailed prediction history
- Store analysis sessions in database

### 🤖 AI Assistant
- Integrated AI chatbot
- Answers water quality related queries
- Assists users with report interpretation

---

## 🛠 Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap
- Jinja2 Templates

### Backend
- Flask
- Python
- SQLAlchemy

### Database
- SQLite

### Machine Learning / Deep Learning
- PyTorch
- NumPy
- Pandas
- Joblib

### Visualization
- Matplotlib

---

## 🧠 Deep Learning Models

The project includes two Deep Learning models:

### 1. Water Potability Prediction
Predicts whether water is:

- Potable
- Non-Potable

### 2. Disease Prediction

Predicts diseases such as:

- Cholera
- Typhoid
- Dysentery
- Hepatitis A
- Fluorosis
- Arsenicosis
- Lead Poisoning
- None (Safe)

---

## 📂 Project Structure

```
Water Quality Analysis
│
├── app.py
├── models.py
├── water.db
│
├── notebooks/
│   ├── dl_potability_model.pt
│   ├── dl_disease_model.pt
│   ├── scalers
│   ├── encoders
│   └── training notebooks
│
├── templates/
│   ├── dashboard.html
│   ├── assistant.html
│   ├── history.html
│   ├── compare.html
│   ├── stats.html
│   └── session_detail.html
│
└── README.md
```

---

## 📊 Input Parameters

The prediction model uses the following water quality parameters:

- pH
- Hardness
- Total Dissolved Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

---

## 📷 Application Screens

Include screenshots like:

- Dashboard
- Prediction Result
- History Page
- Statistics Dashboard
- Comparison Page
- AI Assistant

> Save screenshots inside:

```
images/
```

Example:

```
images/dashboard.png
images/history.png
images/stats.png
images/assistant.png
```

Then use:

```markdown
## Dashboard

![Dashboard](images/dashboard.png)

## Statistics

![Statistics](images/stats.png)

## AI Assistant

![Assistant](images/assistant.png)
```

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/water-quality-analysis.git
```

### Navigate

```bash
cd water-quality-analysis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000/
```

---

## 📈 Future Enhancements

- Cloud Deployment
- User Authentication
- PDF Report Export
- Email Notifications
- Real-time IoT Sensor Integration
- GIS Water Quality Mapping
- Mobile Application
- Multi-language Support

---

## 🎯 Learning Outcomes

This project demonstrates:

- Flask Web Development
- Deep Learning Model Deployment
- SQLAlchemy Database Integration
- PyTorch Model Inference
- Data Visualization
- RESTful Backend Development
- AI-powered Decision Support System

---

## 👨‍💻 Author

**Kaviarasu Subramani**

Computer Science Engineering Student

Interested in:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Full Stack Development

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
