
# COVID-19 New Cases Forecast Model

In this project,the usage of deep learning model to predict the daily new COVID cases were designed based on the dataset gain from year 2020 to 2022. The dataset are saperated into 2 dataset, first dataset for model training while the second dataset for model evaluation or testing.The prediction of new cases (cases_new) in Malaysia will be set to predict based on the past 30 days of number of cases.


## Getting started

Before proceeding to the model development, the data analyses are done based on the section below :-

- Data Loading

  In this section, the data from the CSV files were loaded using Pandas 
  module ```pandas.read_csv``` which convert into dataframe.

- Data Inspection
   
  From the dataframe loaded, the inspection is done to check for any abnormalities on the dataframe which then will be cleaned in the next section. The method used from ```Pandas``` module:

  ```
  df.info()           # Check for the datatype
  df.describe()       # Check for dataframe summary
  ```
- Data Cleaning

  After the inspection, the feature columns, ```cases_new```, were cleaned by converting into numerics using ```numpy``` module and replacing the NaN values exist in the dataframe. 

  ```
  df.isna().sum()         # Check for missing values
  df.duplicated().sum()   # Check for complete duplicates by row
  df.interoplate()        # Fill missing values using interpolation
  np.to_numerics()        # Convert datatype of column to numnerics
  ```
- Data Preprocessing
  
  The data preprocessing function used to process the data before it was fit to the model. In this step, the feature columns were normalize using ```MinMaxScaler```, and then separate based on 30 days window before further split into 4 parts for model training. The processing mainly used ```scikit-learn``` module which in this project, ```sklearn.preprocessing``` and ```sklearn.model_selection```.

  ```
  .fit_transform         # Scale the data into 0 and 1 values
  train_test_split       # Split the data to (x_train,x_test,y_train_y_test)
  ```
## Model development

In this stage, the model was develop using ```LTSM``` neural network which was provided in ```tensorflow``` module.The model architecture can be seen below:

![App Screenshot]()

## Model training

For the model training, the model, which have been explained above will be compile. In this project, the model was compile using Adam as optimizer, Mean Squared Error (MSE) for loss and Mean Absolute Percentage Error (MAPE) as metrics. Then, the model was fit with the (x_train,x_test,y_train_y_test) data split. The result of the training can be seen below:-

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Model evaluation

After completing the model training, the model will need to be evaluate to check if the model prediction is close to the actual dataset. The model evaluation will use the dataset test for evaluation whhich will be concantinate with the previous 30 days from training dataset. The concantinate dataset will then process using the same step of data laoding, inspection, cleaning and preprocessing before the testing dataset can be pass to the model for prediction.This predicion will be then compared with the actual cases by computing the MAPE and MSE. The graph and performance of the model can be see below:-   

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)





## Acknowledgements

 Special thanks to the MOH Malaysia for providing the [data](https://github.com/MoH-Malaysia/covid19-public) on COVID-19 in Malaysia
