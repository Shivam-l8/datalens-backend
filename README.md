# DataLens Backend API

> **Automated Exploratory Data Analysis (EDA) Engine** — A production-ready FastAPI service that performs intelligent data analysis, generates actionable insights, and delivers structured outputs for any tabular dataset.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-orange.svg)](https://pandas.pydata.org/)

---

## 🎯 Project Overview

**DataLens** is a full-stack web application designed to democratize data analysis. This backend service automatically:

- **Detects** column types (numeric, categorical, datetime) without manual configuration
- **Cleans** datasets using explainable, transparent methods
- **Analyzes** relationships, distributions, and trends
- **Generates** human-readable insights and data-driven recommendations
- **Delivers** structured outputs ready for visualization and reporting

This project demonstrates **production-level data engineering**, **statistical analysis**, and **API design** skills relevant to data analyst and business intelligence roles.

---

## 🚀 Key Features

### Automated Data Analysis
- **Schema-agnostic processing**: Works with any CSV or Excel file without predefined schemas
- **Intelligent column detection**: Automatically identifies numeric, categorical, and datetime columns
- **Missing value handling**: Transparent imputation strategies (median for numeric, explicit labels for categorical)
- **Outlier detection**: IQR-based flagging without data loss

### Statistical Analysis
- **Univariate analysis**: Distribution summaries, central tendency, and categorical frequencies
- **Bivariate analysis**: Correlation matrices and relationship identification
- **Time series analysis**: Automatic trend detection when datetime columns are present
- **Segmentation analysis**: Group comparisons across categorical dimensions

### Insight Generation
- **Pattern recognition**: Identifies key drivers and outcome variables
- **Evidence-based insights**: All findings reference actual data patterns
- **Actionable recommendations**: Business-focused suggestions tied to analysis results
- **Plain-language summaries**: Human-readable reports suitable for stakeholders

### API Design
- **RESTful architecture**: Clean, predictable endpoints
- **Type-safe responses**: Pydantic models ensure consistent JSON structure
- **Error handling**: Comprehensive validation and user-friendly error messages
- **Security-first**: No persistent data storage, in-memory processing only

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | FastAPI | Modern, high-performance API framework |
| **Data Processing** | Pandas, NumPy | Data manipulation and numerical computing |
| **Visualization** | Matplotlib, Seaborn | Statistical plotting and chart generation |
| **Validation** | Pydantic | Request/response schema validation |
| **Server** | Uvicorn | ASGI server for production deployment |

---

## 📋 Prerequisites

- **Python 3.9+**
- **pip** (Python package manager)
- **Virtual environment** (recommended)

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd datalens-backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Development Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

---

## 📡 API Endpoints

### `POST /analyze`

Upload a dataset and receive comprehensive analysis results.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: File upload (CSV or Excel, max 10MB)

**Response:**
```json
{
  "summary": {
    "dataset_shape": {"rows": 1000, "columns": 15},
    "column_overview": {...},
    "missing_summary": {...},
    "detected_roles": {
      "numeric": ["price", "quantity"],
      "categorical": ["category", "region"],
      "datetime": ["date"]
    }
  },
  "summary_statistics": {...},
  "correlations": {...},
  "key_insights": [
    {
      "title": "Strong correlation detected",
      "detail": "Price and quantity show a negative correlation of -0.72..."
    }
  ],
  "recommendations": [
    {
      "title": "Optimize pricing strategy",
      "action": "Consider dynamic pricing models to balance revenue and volume..."
    }
  ],
  "charts": [...],
  "cleaned_csv_base64": "...",
  "insight_text": "Full narrative summary..."
}
```

### `GET /health`

Health check endpoint for monitoring and deployment verification.

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🏗️ Project Structure

```
datalens-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and route handlers
│   ├── eda.py               # Core EDA and insight generation logic
│   └── schemas.py           # Pydantic models for request/response validation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Modules

- **`main.py`**: API routes, file upload handling, and error management
- **`eda.py`**: Reusable functions for data cleaning, statistical analysis, and insight generation
- **`schemas.py`**: Type-safe data models ensuring API contract consistency

---

## 🔍 Analysis Pipeline

1. **File Validation**: Type and size checks
2. **Data Loading**: Pandas-based CSV/Excel parsing
3. **Column Role Detection**: Automatic identification of numeric, categorical, and datetime columns
4. **Data Cleaning**: Duplicate removal, missing value imputation, outlier flagging
5. **Statistical Analysis**: Summary statistics, correlations, distributions
6. **Visualization Data Generation**: Chart-ready JSON structures
7. **Insight Generation**: Pattern recognition and recommendation synthesis
8. **Response Assembly**: Structured JSON with all analysis outputs

---

## 🚢 Deployment

### Option 1: Railway / Render / Fly.io

1. **Create account** on your preferred platform
2. **Connect repository** to the platform
3. **Set build command**: `pip install -r requirements.txt`
4. **Set start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Configure environment variables** (if needed)
6. **Deploy**

### Option 2: Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t datalens-backend .
docker run -p 8000:8000 datalens-backend
```

### Option 3: AWS / GCP / Azure

- Use **AWS Lambda** with API Gateway (serverless)
- Use **Google Cloud Run** (containerized)
- Use **Azure App Service** (PaaS)

**Note**: Update CORS origins in `main.py` to include your frontend domain after deployment.

---

## 🧪 Testing

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Analyze dataset
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@/path/to/your/dataset.csv"
```

### Using Swagger UI

Navigate to `http://localhost:8000/docs` and use the interactive API explorer.

---

## 📊 Skills Demonstrated

This project showcases:

- ✅ **Data Engineering**: ETL pipelines, data cleaning, schema inference
- ✅ **Statistical Analysis**: Descriptive statistics, correlation analysis, trend detection
- ✅ **API Development**: RESTful design, type safety, error handling
- ✅ **Python Proficiency**: Object-oriented design, pandas/numpy expertise
- ✅ **Production Practices**: Security considerations, validation, documentation
- ✅ **Business Acumen**: Insight generation, actionable recommendations

---

## 🔒 Security & Privacy

- **No persistent storage**: All data processed in-memory only
- **No logging of datasets**: Only error messages logged, never data content
- **File size limits**: 10MB maximum to prevent resource exhaustion
- **Input validation**: Strict file type and structure checks
- **CORS configuration**: Restricted to authorized frontend origins

---

## 🐛 Troubleshooting

### Import Errors
- Ensure virtual environment is activated
- Verify all dependencies installed: `pip install -r requirements.txt`

### Port Already in Use
- Change port: `uvicorn app.main:app --port 8001`
- Or kill existing process using port 8000

### Excel File Errors
- Ensure `openpyxl` is installed: `pip install openpyxl`

---

## 📝 License

This project is part of a portfolio for educational and professional purposes.

---

## 👤 Author

**Shivam**  
*Data Analyst | Business Intelligence Enthusiast*



---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation library
- [Seaborn](https://seaborn.pydata.org/) - Statistical visualization

---

**Ready to analyze your data?** Start the server and visit `/docs` to explore the API interactively!

