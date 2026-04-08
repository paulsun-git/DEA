# DEA
## Project Introduction
This project is an EDA (Exploratory Data Analysis) project, designed to mine the hidden patterns, abnormal features and variable correlations behind the dataset through a series of data processing, visualization and statistical analysis operations, so as to provide reliable data support for subsequent modeling and decision-making. Using Python as the main development language, combined with commonly used data analysis libraries such as pandas, numpy, matplotlib and seaborn, the project realizes the full-process automation from data loading, cleaning, preprocessing to visualization analysis, which is both easy to use and scalable, and suitable for rapid exploratory analysis of various structured datasets.
The schematic diagram of the core process of the project is as follows:
![EDA Project Core Process](images/eda_flow.png)<!-- The storage location of the sample image is explained in detail in the "Project Structure and Usage" section below -->
## Dataset
The dataset used in this project is 【Specific Dataset Name, e.g.: A Certain Industry User Behavior Dataset/iris Dataset/House Price Prediction Dataset】. This dataset is mainly used for 【Explain the purpose of the dataset, e.g.: Analyzing user consumption habits, verifying the effect of classification algorithms, exploring factors affecting house prices】, including 【Number】 samples and 【Number】 feature variables, covering various data types such as 【Feature Types, e.g.: Numerical, Categorical, Time Series】.
### Dataset Source
The dataset is from a public data platform. Specific link: [Dataset Source Link](https://xxx.xxx.xxx) (e.g., Kaggle, UCI Machine Learning Repository, domestic public data platforms, etc., replace with the actual link). You can directly download the original dataset through the link, or automatically pull it through the script provided by the project.
### Data Processing Methods
1. Data Loading: Use pandas to read the original dataset (supporting csv, xlsx, txt and other formats), and handle problems such as encoding anomalies and path anomalies;
2. Data Cleaning: Eliminate features with excessively high missing value ratio (e.g., missing rate > 30%), fill the remaining missing values with mean/median/mode (for numerical types) and .mode() (for categorical types); Handle outliers, identify abnormal samples through box plots and Z-score methods, and choose to eliminate or correct them according to business scenarios;
3. Data Preprocessing: Encode categorical variables (One-Hot Encoding, Label Encoding), standardize/normalize numerical variables to eliminate the impact of dimensionality;
## Project Structure and Usage
The following is a description of the core directory of the GitHub project. All directories and files are classified by function for easy reference and maintenance:
EDA/                     # Project root directory
├── data/                # Dataset directory (stores raw data and processed data)
│   ├── raw/             # Raw dataset (put here after downloading from the source link)
│   └── processed/       # Processed data (cleaned and preprocessed dataset for analysis)
├── images/              # Image storage directory (Key point: put sample images here)
│   └── eda_flow.png     # Sample image (project core process diagram, can be replaced with your model image)
├── src/                 # Core code directory
│   ├── data_loader.py   # Data loading script (reads raw data in the data/raw directory)
│   ├── data_preprocess.py # Data preprocessing script (implements cleaning, encoding, standardization and other operations)
│   ├── eda_analysis.py  # EDA analysis script (statistical analysis, feature correlation analysis)
│   └── visualization.py # Visualization script (generates charts and saves them to the images directory)
├── requirements.txt     # Dependent library list (Python libraries required by the project and their corresponding versions)
└── README.md            # Project description document (current document)


