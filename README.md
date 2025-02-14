# Weather Prediction Model
## Predict maximum temperature for a day based on historical weather data.

## Overview
This project implements a weather prediction model using machine learning techniques to forecast maximum temperatures based on historical weather data.

## Features
- Data preprocessing and cleaning
- Feature engineering with rolling averages and percentage differences
- Ridge regression model for temperature prediction
- Visualization of weather patterns and prediction errors
- Model evaluation using MAE and MSE metrics

## Requirements
- Python 3.x
- Required Python packages:
  - pandas
  - matplotlib
  - scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/patelritiq/Weather-Prediction-Model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Weather-Prediction-Model
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your weather data in a CSV file named `weather.csv`
2. Run the prediction model:
   ```python
   python maxtempweatherpredict.py
   ```
3. The script will:
   - Load and preprocess the data
   - Train the prediction model
   - Generate visualizations
   - Display model performance metrics

## Model Details
- **Algorithm**: Ridge Regression
- **Features**: Rolling averages, percentage differences, monthly and daily averages
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

## Visualizations
The model generates the following plots:
1. Snow depth over years
2. Distribution of prediction errors
3. Various weather pattern visualizations

## Future Enhancements
- Incorporate additional weather features
- Implement more advanced machine learning models
- Add real-time weather data integration
- Develop a web-based interface for predictions

## Author
Ritik Pratap Singh Patel  
Data Science Intern at Zidio Development  
Completion Date: 07 May 2024

## License
This project is licensed under the MIT License - see the LICENSE file for details.
