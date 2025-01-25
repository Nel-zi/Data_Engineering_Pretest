# Data_Engineering_Pretest
 
 ### **Documentation and Setup Instructions**

This documentation provides guidance on setting up, running, and understanding the codebase, including instructions for installing dependencies, configuring the environment, and running the analysis.

---

### **1. Prerequisites**

Before proceeding, ensure you have the following installed:
- **Python** (version 3.8 or higher)
- **PostgreSQL** (for database integration)
- **Git** (if cloning the repository)
- A code editor or IDE (e.g., VS Code, PyCharm)

---

### **2. Directory Structure**
Here is an example of how the project directory is organized:

```
project-directory/
│
├── data/
│   ├── items.csv
│   ├── promotion.csv
│   ├── sales.csv
│   ├── supermarkets.csv
│
├── src/
│   ├── main.py         # Main script for running analysis
│   ├── database.py     # Handles database operations
│   ├── data_cleaning.py # Functions for cleaning datasets
│   ├── analysis.py     # Business insights logic
│   └── visualization.py # Visualization functions
│
├── requirements.txt    # List of dependencies
├── .env                # Environment variables (e.g., database credentials)
├── README.md           # Project overview and usage
└── docs/
    └── report.pdf      # Final report detailing tasks and insights
```

---

### **3. Setting Up the Project**

#### **Step 1: Clone the Repository**
If the project is hosted on a Git repository, clone it:
```bash
git clone <repository-url>
cd project-directory
```

#### **Step 2: Create a Virtual Environment**
Set up a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

#### **Step 3: Install Dependencies**
Install the required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### **Step 4: Configure Environment Variables**
Create a `.env` file (if it doesn’t already exist) in the project root. Add the following variables:
```
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=your_host
DB_PORT=your_port
```

Replace placeholders with your actual PostgreSQL credentials.

#### **Step 5: Load Data into PostgreSQL**
Use the `database.py` script to upload cleaned datasets to the PostgreSQL database:
```bash
python src/database.py
```
This script:
- Connects to the PostgreSQL database using credentials in `.env`.
- Creates tables for items, promotions, sales, and supermarkets.
- Loads the data from the `data/` folder into the respective tables.

---

### **4. Running the Code**

#### **Step 1: Perform Data Cleaning**
Run the data cleaning script to preprocess and clean the datasets:
```bash
python src/data_cleaning.py
```

#### **Step 2: Generate Business Insights**
Use the `analysis.py` script to analyze branch-level sales patterns and promotion effectiveness:
```bash
python src/analysis.py
```

#### **Step 3: Visualize Results**
Run the visualization script to create charts and heatmaps for promotion effectiveness and sales trends:
```bash
python src/visualization.py
```

#### **Step 4: Run the Main Script**
Alternatively, you can run the `main.py` script, which combines all the steps:
```bash
python src/main.py
```

---

### **5. Key Configuration Files**

#### **requirements.txt**
This file lists all Python dependencies. Here’s an example:
```
pandas
numpy
matplotlib
seaborn
sqlalchemy
psycopg2-binary
python-dotenv
```

Install these by running:
```bash
pip install -r requirements.txt
```

#### **.env**
Holds sensitive credentials (e.g., database information). Example:
```
DB_NAME=supermarket_data
DB_USER=admin
DB_PASSWORD=securepassword
DB_HOST=localhost
DB_PORT=5432
```

---

### **6. Key Features of the Code**

1. **Data Cleaning**:
   - Handles missing values, duplicates, and data type corrections.

2. **Database Integration**:
   - Uploads cleaned data into a PostgreSQL database for centralized storage and analysis.

3. **Business Analysis**:
   - Generates actionable insights, such as branch-level sales patterns and promotion effectiveness.

4. **Visualization**:
   - Provides visual insights through heatmaps, bar plots, and other charts.

---

### **7. Troubleshooting**

1. **Dependency Issues**:
   - Ensure you’re using the correct Python version.
   - If errors occur during installation, update `pip`:
     ```bash
     pip install --upgrade pip
     ```

2. **Database Connection Errors**:
   - Verify that the PostgreSQL server is running and the `.env` file contains the correct credentials.

3. **Data File Issues**:
   - Ensure all required CSV files are in the `data/` folder. Missing files will cause errors.

---

### **8. Notes for Future Development**
1. Add year information to the sales dataset for temporal analysis.
2. Include additional validation steps for ensuring data quality during extraction.