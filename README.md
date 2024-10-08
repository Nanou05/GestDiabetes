# GestDiabetes
### Analysis of Web data on Gestational Diabetes

## Introduction
* The objective of this project is to analyze social media exchanges about gestational diabetes.
* From these exchanges, The symptoms will be identified and extracted.
* The focus is mainly on pregnant women consuming either insulin, metformin or both.

## Data source
* the data is extracted from 2 different exchange forums online.
* The extracted data is not provided in this repo, but can be sent if requested.

## Scripts
* This repo is composed of 3 main python sripts to run in this order: (available in "Code" branch)
* 1 : web_scrapping.py: Contains the web scraping scripts to collect data from various social media platforms and forums.
* 2 : custom_contractions.py: A Python script for handling custom English contractions in the text data.
* contractions.json: A JSON file with common contractions used in social media posts, generated by running the code file custom_contractions.py
* 3 : data_preprocessing.py: Scripts for cleaning and preprocessing the raw data.
* 4 : visualisation_analysis.py: Contains scripts for visualizing the data and analysis results.

## let's get started

### Prerequisites

- Python 3.8
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/nanou05/GestDiabetes.git
    ```
2. Navigate to the project directory:
    ```bash
    cd GestDiabetes
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
## Usage 

1. **Data Collection**: Run `web_scrapping.py`
    ```bash
    python web_scrapping.py
    ```
2. **Data Preprocessing**: Use `data_preprocessing.py`
    ```bash
    python data_preprocessing.py
    ```
3. **Custom Contractions Handling**: `custom_contractions.py` 
    ```bash
    python custom_contractions.py
    ```
4. **Visualization and Analysis**: Run `Visualisation_analysis.py` 
    ```bash
    python Visualisation_analysis.py
    ```
    
## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.
    
