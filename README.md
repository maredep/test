# SP901 Capstone Project

# 5.3 Final Project

## 5.3.1 Python for Data Science and Machine Learning

Guidelines:
1. Using the dataset provided, create a binary classification
2. Perform the following:
    *  Perform an exploratory analysis (scaling, PCA, unbalanced)
    *  Split the data (train, validation, test)
    *  Perform 10-fold cross-validation and grid search (method of your choice)
    *  Compare the different classification methods
        (Logistic Regression, KNN, SVM, RF, XGBOOST)
    *  Show evaluation metrics (ROC-AUC, accuracy, f-1 score)
    *  Submit the jupyter notebook

### Importing Libraries


```python
# Importing necessary libraries
# For performing linear algebra or computing
import numpy as np 
 
# For data processing
import pandas as pd
 
# For visualisation
import matplotlib.pyplot as plt
import seaborn as sns
```

### Loading Dataset


```python
# Load dataset
data = pd.read_csv("C:/Users/mneme/Desktop/SP901_CS_completedata.csv", sep=';')
```


```python
# Creating a list of Result
compare_classification_methods = []
```

### Data Preprocessing


```python
# Check for missing values
data.isnull().sum()
```




    PatientID             0
    Failure.binary        0
    Entropy_cooc.W.ADC    0
    GLNU_align.H.PET      0
    Min_hist.PET          0
                         ..
    GLNU_norm.W.ADC       0
    ZSNU_norm.W.ADC       0
    GLVAR_area.W.ADC      0
    ZSVAR.W.ADC           0
    Entropy_area.W.ADC    0
    Length: 430, dtype: int64




```python
# Check all column data types
data.dtypes
```




    PatientID               int64
    Failure.binary          int64
    Entropy_cooc.W.ADC    float64
    GLNU_align.H.PET      float64
    Min_hist.PET          float64
                           ...   
    GLNU_norm.W.ADC       float64
    ZSNU_norm.W.ADC       float64
    GLVAR_area.W.ADC      float64
    ZSVAR.W.ADC           float64
    Entropy_area.W.ADC    float64
    Length: 430, dtype: object




```python
# Getting some information about the dataset
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 197 entries, 0 to 196
    Columns: 430 entries, PatientID to Entropy_area.W.ADC
    dtypes: float64(428), int64(2)
    memory usage: 661.9 KB
    


```python
# Check initial data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientID</th>
      <th>Failure.binary</th>
      <th>Entropy_cooc.W.ADC</th>
      <th>GLNU_align.H.PET</th>
      <th>Min_hist.PET</th>
      <th>Max_hist.PET</th>
      <th>Mean_hist.PET</th>
      <th>Variance_hist.PET</th>
      <th>Standard_Deviation_hist.PET</th>
      <th>Skewness_hist.PET</th>
      <th>...</th>
      <th>LZLGE.W.ADC</th>
      <th>LZHGE.W.ADC</th>
      <th>GLNU_area.W.ADC</th>
      <th>ZSNU.W.ADC</th>
      <th>ZSP.W.ADC</th>
      <th>GLNU_norm.W.ADC</th>
      <th>ZSNU_norm.W.ADC</th>
      <th>GLVAR_area.W.ADC</th>
      <th>ZSVAR.W.ADC</th>
      <th>Entropy_area.W.ADC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>12.85352</td>
      <td>46.256345</td>
      <td>6.249117</td>
      <td>17.825541</td>
      <td>9.783773</td>
      <td>6.814365</td>
      <td>2.612479</td>
      <td>0.688533</td>
      <td>...</td>
      <td>0.00690</td>
      <td>6201.93480</td>
      <td>4.13400</td>
      <td>239.28938</td>
      <td>0.97918</td>
      <td>0.01899</td>
      <td>0.95586</td>
      <td>1145.10496</td>
      <td>0.02586</td>
      <td>6.28632</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>12.21115</td>
      <td>27.454540</td>
      <td>11.005214</td>
      <td>26.469077</td>
      <td>15.426640</td>
      <td>12.932074</td>
      <td>3.598298</td>
      <td>0.789526</td>
      <td>...</td>
      <td>0.00423</td>
      <td>16054.01263</td>
      <td>8.37627</td>
      <td>644.73702</td>
      <td>0.95637</td>
      <td>0.01461</td>
      <td>0.93288</td>
      <td>847.52537</td>
      <td>0.04153</td>
      <td>6.77853</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>12.75682</td>
      <td>90.195696</td>
      <td>2.777718</td>
      <td>6.877486</td>
      <td>4.295330</td>
      <td>0.923425</td>
      <td>0.962163</td>
      <td>0.248637</td>
      <td>...</td>
      <td>0.00453</td>
      <td>6674.63840</td>
      <td>13.11686</td>
      <td>1165.70261</td>
      <td>0.97268</td>
      <td>0.02501</td>
      <td>0.91537</td>
      <td>1923.85705</td>
      <td>0.07104</td>
      <td>7.15685</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>13.46730</td>
      <td>325.643330</td>
      <td>6.296588</td>
      <td>22.029843</td>
      <td>10.334779</td>
      <td>6.649795</td>
      <td>2.580759</td>
      <td>0.832011</td>
      <td>...</td>
      <td>0.00888</td>
      <td>17172.90951</td>
      <td>23.84726</td>
      <td>2760.41293</td>
      <td>0.97203</td>
      <td>0.01069</td>
      <td>0.94658</td>
      <td>1329.95290</td>
      <td>0.03848</td>
      <td>7.29521</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>12.63733</td>
      <td>89.579042</td>
      <td>3.583846</td>
      <td>7.922501</td>
      <td>4.454175</td>
      <td>0.572094</td>
      <td>0.757225</td>
      <td>1.574845</td>
      <td>...</td>
      <td>0.00405</td>
      <td>13231.94294</td>
      <td>8.14437</td>
      <td>784.59729</td>
      <td>0.96469</td>
      <td>0.02526</td>
      <td>0.93769</td>
      <td>1116.38669</td>
      <td>0.05223</td>
      <td>7.05149</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 430 columns</p>
</div>




```python
# We drop PatientID cause it is not relevant feature to our analysis 
columns_to_drop = ['PatientID']
data = data.drop(columns=columns_to_drop)
data.shape
```




    (197, 429)




```python
# Basic summary statistics
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Failure.binary</th>
      <th>Entropy_cooc.W.ADC</th>
      <th>GLNU_align.H.PET</th>
      <th>Min_hist.PET</th>
      <th>Max_hist.PET</th>
      <th>Mean_hist.PET</th>
      <th>Variance_hist.PET</th>
      <th>Standard_Deviation_hist.PET</th>
      <th>Skewness_hist.PET</th>
      <th>Kurtosis_hist.PET</th>
      <th>...</th>
      <th>LZLGE.W.ADC</th>
      <th>LZHGE.W.ADC</th>
      <th>GLNU_area.W.ADC</th>
      <th>ZSNU.W.ADC</th>
      <th>ZSP.W.ADC</th>
      <th>GLNU_norm.W.ADC</th>
      <th>ZSNU_norm.W.ADC</th>
      <th>GLVAR_area.W.ADC</th>
      <th>ZSVAR.W.ADC</th>
      <th>Entropy_area.W.ADC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>...</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.340102</td>
      <td>12.278600</td>
      <td>95.381938</td>
      <td>8.513255</td>
      <td>24.271413</td>
      <td>13.008133</td>
      <td>9.257452</td>
      <td>3.049220</td>
      <td>0.911980</td>
      <td>0.490932</td>
      <td>...</td>
      <td>0.006405</td>
      <td>13333.581481</td>
      <td>40.154389</td>
      <td>3334.075705</td>
      <td>1.193836</td>
      <td>0.016572</td>
      <td>1.157634</td>
      <td>1114.711636</td>
      <td>0.065497</td>
      <td>8.507117</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.474950</td>
      <td>1.039816</td>
      <td>86.089059</td>
      <td>4.985543</td>
      <td>14.779666</td>
      <td>7.668180</td>
      <td>9.303475</td>
      <td>1.848637</td>
      <td>0.691920</td>
      <td>3.041625</td>
      <td>...</td>
      <td>0.032306</td>
      <td>9140.346577</td>
      <td>52.092487</td>
      <td>4751.131998</td>
      <td>0.423532</td>
      <td>0.031414</td>
      <td>0.412787</td>
      <td>755.908819</td>
      <td>0.056252</td>
      <td>2.995206</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>9.532740</td>
      <td>9.445031</td>
      <td>1.484508</td>
      <td>4.164474</td>
      <td>2.424636</td>
      <td>0.178752</td>
      <td>0.419449</td>
      <td>-0.001136</td>
      <td>-2.266122</td>
      <td>...</td>
      <td>-0.062616</td>
      <td>1369.130190</td>
      <td>2.015900</td>
      <td>84.039160</td>
      <td>0.851807</td>
      <td>-0.054262</td>
      <td>0.792028</td>
      <td>253.629375</td>
      <td>-0.029824</td>
      <td>5.585010</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
      <td>11.558840</td>
      <td>37.518193</td>
      <td>5.151990</td>
      <td>13.071684</td>
      <td>7.497794</td>
      <td>2.258260</td>
      <td>1.639108</td>
      <td>0.444828</td>
      <td>-0.525860</td>
      <td>...</td>
      <td>-0.011160</td>
      <td>6881.763841</td>
      <td>9.340283</td>
      <td>741.277380</td>
      <td>0.945840</td>
      <td>0.001476</td>
      <td>0.908540</td>
      <td>564.917867</td>
      <td>0.031800</td>
      <td>6.626250</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>12.278790</td>
      <td>80.034684</td>
      <td>7.388754</td>
      <td>21.013614</td>
      <td>11.449486</td>
      <td>6.450421</td>
      <td>2.734120</td>
      <td>0.734796</td>
      <td>-0.167186</td>
      <td>...</td>
      <td>0.009070</td>
      <td>11685.594830</td>
      <td>20.363374</td>
      <td>1479.035520</td>
      <td>0.966065</td>
      <td>0.018532</td>
      <td>0.938043</td>
      <td>983.073750</td>
      <td>0.055972</td>
      <td>7.025632</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.000000</td>
      <td>12.977330</td>
      <td>112.145185</td>
      <td>11.005214</td>
      <td>33.761142</td>
      <td>17.386702</td>
      <td>12.682440</td>
      <td>4.209453</td>
      <td>1.199956</td>
      <td>0.501737</td>
      <td>...</td>
      <td>0.021579</td>
      <td>17172.909510</td>
      <td>48.480280</td>
      <td>3976.605794</td>
      <td>1.797414</td>
      <td>0.033476</td>
      <td>1.677856</td>
      <td>1295.180470</td>
      <td>0.091940</td>
      <td>11.170020</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>14.510471</td>
      <td>559.351571</td>
      <td>28.404496</td>
      <td>79.985858</td>
      <td>44.043168</td>
      <td>49.012054</td>
      <td>9.929300</td>
      <td>4.901172</td>
      <td>33.742118</td>
      <td>...</td>
      <td>0.136980</td>
      <td>51885.362160</td>
      <td>387.348504</td>
      <td>35037.698160</td>
      <td>1.980520</td>
      <td>0.086040</td>
      <td>2.007120</td>
      <td>4306.766300</td>
      <td>0.318752</td>
      <td>15.380880</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 429 columns</p>
</div>




```python
# Univariate Analysis
# Histogram of All Features
# Exclude the first column ['Failure.binary'] since its a binary
columns = data.columns[1:]  

# Visualize the first columns
for i in range(0, 10, 10):
    plt.figure(figsize=(15, 6))
    subset = columns[i:i+10]
    for idx, col in enumerate(subset):
        plt.subplot(2, 5, idx + 1)
        sns.distplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
```


    
![png](README_files/README_16_0.png)
    


Based on the statistics and the distplot above, the dataset has a lot of outlier

#### Outlier Removal


```python
# Defining a Z-score threshold
threshold = 3

# Calculate Z-scores for each column in the DataFrame
z_scores = np.abs((data - data.mean()) / data.std())

# Filter data based on the threshold
data = data[(z_scores < threshold).all(axis=1)]
print(data.shape)
data.describe()
```

    (118, 429)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Failure.binary</th>
      <th>Entropy_cooc.W.ADC</th>
      <th>GLNU_align.H.PET</th>
      <th>Min_hist.PET</th>
      <th>Max_hist.PET</th>
      <th>Mean_hist.PET</th>
      <th>Variance_hist.PET</th>
      <th>Standard_Deviation_hist.PET</th>
      <th>Skewness_hist.PET</th>
      <th>Kurtosis_hist.PET</th>
      <th>...</th>
      <th>LZLGE.W.ADC</th>
      <th>LZHGE.W.ADC</th>
      <th>GLNU_area.W.ADC</th>
      <th>ZSNU.W.ADC</th>
      <th>ZSP.W.ADC</th>
      <th>GLNU_norm.W.ADC</th>
      <th>ZSNU_norm.W.ADC</th>
      <th>GLVAR_area.W.ADC</th>
      <th>ZSVAR.W.ADC</th>
      <th>Entropy_area.W.ADC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>...</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.330508</td>
      <td>12.265963</td>
      <td>85.673275</td>
      <td>7.657145</td>
      <td>20.798990</td>
      <td>11.633292</td>
      <td>8.083566</td>
      <td>2.650078</td>
      <td>0.724525</td>
      <td>0.085722</td>
      <td>...</td>
      <td>-0.000452</td>
      <td>10673.274099</td>
      <td>27.851352</td>
      <td>2233.416167</td>
      <td>1.006545</td>
      <td>0.009333</td>
      <td>0.974908</td>
      <td>882.389337</td>
      <td>0.049369</td>
      <td>7.183618</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.472402</td>
      <td>0.948230</td>
      <td>65.624263</td>
      <td>3.753674</td>
      <td>10.194541</td>
      <td>5.773281</td>
      <td>6.595905</td>
      <td>1.295172</td>
      <td>0.413204</td>
      <td>0.953426</td>
      <td>...</td>
      <td>0.028094</td>
      <td>6044.745330</td>
      <td>27.037457</td>
      <td>2320.619464</td>
      <td>0.237401</td>
      <td>0.027976</td>
      <td>0.232854</td>
      <td>476.807263</td>
      <td>0.040516</td>
      <td>1.637960</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>9.780640</td>
      <td>13.658399</td>
      <td>2.063546</td>
      <td>4.481790</td>
      <td>3.108424</td>
      <td>0.346084</td>
      <td>0.590950</td>
      <td>-0.000568</td>
      <td>-1.427184</td>
      <td>...</td>
      <td>-0.061976</td>
      <td>1369.130190</td>
      <td>2.374870</td>
      <td>91.949230</td>
      <td>0.851807</td>
      <td>-0.054262</td>
      <td>0.792028</td>
      <td>253.629375</td>
      <td>-0.029824</td>
      <td>5.585010</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
      <td>11.660185</td>
      <td>32.735897</td>
      <td>4.446142</td>
      <td>12.999408</td>
      <td>6.256104</td>
      <td>2.344858</td>
      <td>1.535263</td>
      <td>0.390722</td>
      <td>-0.437609</td>
      <td>...</td>
      <td>-0.013075</td>
      <td>6010.911250</td>
      <td>9.340808</td>
      <td>742.929782</td>
      <td>0.934242</td>
      <td>-0.000229</td>
      <td>0.896453</td>
      <td>488.975665</td>
      <td>0.024300</td>
      <td>6.455271</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>12.270140</td>
      <td>68.043387</td>
      <td>6.888983</td>
      <td>20.360035</td>
      <td>11.041045</td>
      <td>6.673200</td>
      <td>2.662501</td>
      <td>0.683343</td>
      <td>-0.152004</td>
      <td>...</td>
      <td>0.005505</td>
      <td>9699.487420</td>
      <td>18.461570</td>
      <td>1336.865905</td>
      <td>0.956414</td>
      <td>0.013600</td>
      <td>0.929805</td>
      <td>863.279430</td>
      <td>0.052332</td>
      <td>6.870535</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1.000000</td>
      <td>12.966412</td>
      <td>113.313846</td>
      <td>9.871236</td>
      <td>26.018870</td>
      <td>14.989738</td>
      <td>11.419836</td>
      <td>3.389477</td>
      <td>0.987212</td>
      <td>0.327711</td>
      <td>...</td>
      <td>0.016723</td>
      <td>15205.102668</td>
      <td>38.341445</td>
      <td>3146.816648</td>
      <td>0.971985</td>
      <td>0.025260</td>
      <td>0.946587</td>
      <td>1134.343272</td>
      <td>0.067544</td>
      <td>7.131410</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>14.465471</td>
      <td>325.643330</td>
      <td>16.983672</td>
      <td>42.966732</td>
      <td>27.033122</td>
      <td>24.046560</td>
      <td>6.962134</td>
      <td>1.951428</td>
      <td>4.065725</td>
      <td>...</td>
      <td>0.068490</td>
      <td>32610.285360</td>
      <td>193.674252</td>
      <td>15648.256480</td>
      <td>1.969320</td>
      <td>0.073456</td>
      <td>1.915760</td>
      <td>2743.237630</td>
      <td>0.164320</td>
      <td>14.610520</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 429 columns</p>
</div>




```python
# Histogram of All Features after Removal of Outlier
# Exclude the first column ['Failure.binary'] since its a binary
columns = data.columns[1:]  

# Visualize the first 10 columns 
for i in range(0, 10, 10):
    plt.figure(figsize=(15, 6))
    subset = columns[i:i+10]
    for idx, col in enumerate(subset):
        plt.subplot(2, 5, idx + 1)
        sns.distplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
```


    
![png](README_files/README_20_0.png)
    


#### Imbalance Dataset


```python
# Check class distribution
class_distribution = data['Failure.binary'].value_counts()
print(class_distribution)
```

    0    79
    1    39
    Name: Failure.binary, dtype: int64
    


```python
import sklearn
print(sklearn.__version__)
```

    1.0.2
    


```python
# Splitting data to Independent and Dependent Variable (Target)
X = data.drop('Failure.binary', axis=1)  
y = data['Failure.binary'] # 'Failure.binary' is the target
```


```python
import matplotlib.pyplot as plt

plt.bar(['0', '1'], class_distribution, color=['blue', 'orange'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Failure Distribution')
plt.show()
```


    
![png](README_files/README_25_0.png)
    


The distribution shows that there is an imbalance on our dataset. We will have a problem if we used it especially when we will be using 10 fold cross-validation that requires a lot of data as well as on the grid search. It also can affect our model performance.

#### SMOTE for Handling Imbalance Dataset


```python
# Installing imbalanced learn
# It is an old version to make sure that it is compatible to my system
# You may upgrade to the latest version
!pip install imbalanced-learn==0.8.0 --user
```

    Requirement already satisfied: imbalanced-learn==0.8.0 in c:\users\mneme\appdata\roaming\python\python37\site-packages (0.8.0)
    Requirement already satisfied: numpy>=1.13.3 in c:\users\mneme\appdata\roaming\python\python37\site-packages (from imbalanced-learn==0.8.0) (1.16.5)
    Requirement already satisfied: scipy>=0.19.1 in c:\users\mneme\appdata\roaming\python\python37\site-packages (from imbalanced-learn==0.8.0) (1.7.3)
    Requirement already satisfied: scikit-learn>=0.24 in c:\programdata\anaconda3\lib\site-packages (from imbalanced-learn==0.8.0) (1.0.2)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from imbalanced-learn==0.8.0) (0.13.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn>=0.24->imbalanced-learn==0.8.0) (3.1.0)
    


```python
from imblearn.over_sampling import SMOTE
from collections import Counter
# Show the class distribution before applying SMOTE
print("Before SMOTE:", Counter(y))

# SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Show the class distribution after applying SMOTE
print("After SMOTE:", Counter(y))
```

    Before SMOTE: Counter({0: 79, 1: 39})
    After SMOTE: Counter({0: 79, 1: 79})
    

We use SMOTE to deal with imbalance on our minority class. Now, we have a balanced dataset. We could now start our analysis.

### Exploratory Data Analysis

#### Scaling using StandardScaler


```python
# We will scale our data using standardscaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
data_scaled = scaler.fit_transform(X)

# Convert scaled data arrays back to dataframes
data_scaled_df = pd.DataFrame(data_scaled, columns=X.columns)
data_scaled_df.shape
```




    (158, 428)



#### Dimension Reduction

A dataset with samples (rows) < features (columns), it qualifies as a high-dimensional dataset requiring dimensionality reduction. There are lots of methods but our plan is to employ either PCA or RFE methods for this project. 

##### Sctterplot Diagram of 2 PCA Selected Components and 2 RFE Selected Features 


```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Apply PCA for dimensionality reduction to 30 components
pca = PCA(n_components=30)
X_pca = pca.fit_transform(data_scaled_df)

# Apply RFE for dimensionality reduction to 30 features
model = LogisticRegression(solver='liblinear', random_state = 42, max_iter = 1000)
rfe = RFE(model, n_features_to_select=30)
X_rfe = rfe.fit_transform(data_scaled_df, y)

# Visualize PCA
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("PCA")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Visualize RFE
plt.subplot(1, 2, 2)
plt.title("RFE")
plt.scatter(X_rfe[:, 0], X_rfe[:, 1], c=y, cmap='coolwarm')
plt.xlabel('Selected Feature 1')
plt.ylabel('Selected Feature 2')

plt.tight_layout()
plt.show()

```


    
![png](README_files/README_37_0.png)
    


##### Variance of PCA Selected Components


```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=30)
pca.fit(data_scaled_df)
reduced_data = pca.transform(data_scaled_df)

# Analyze results - For example, print the explained variance ratio
#print("Explained variance ratio:", pca.explained_variance_ratio_)

total_variance_sklearn = np.sum(pca.explained_variance_ratio_)
print("Total variance explained (using sklearn PCA):", total_variance_sklearn)
# Perform further analysis or visualization with 'reduced_data'

num_components = 30  # Number of principal components you want to include
# Create a DataFrame from the reduced data
reduced_df = pd.DataFrame(data=reduced_data[:, :num_components], columns=[f'PC{i}' for i in range(1, num_components + 1)])
```

    Total variance explained (using sklearn PCA): 0.9871592909127157
    


```python
explained_var_ratio = pca.explained_variance_ratio_

# Plotting the explained variance ratios
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio, alpha=0.5, align='center')
plt.step(range(1, len(explained_var_ratio) + 1), np.cumsum(explained_var_ratio), where='mid')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Each Principal Component')
plt.show()
```


    
![png](README_files/README_40_0.png)
    


The analysis demonstrates that by utilizing 30 features, we can encompass 98.7% of the dataset. Therefore, our project will operate with these 30 selected features.

Checking PCA performance


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(reduced_df, y, stratify=y, test_size=0.3, random_state=42)

logreg = LogisticRegression(solver='liblinear', random_state = 42, max_iter = 1000)
print("PCA Accuracy: {} ".format(logreg.fit(data_scaled_df, y).score(data_scaled_df, y)))
```

    PCA Accuracy: 0.9810126582278481 
    

#### Feature Selection using RFE


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
logreg = LogisticRegression(solver='liblinear', random_state = 42, max_iter = 1000)

# Initialize RFE with logistic regression as the estimator
# Set the number of features you want to select (e.g., 5 in this case)
num_features_to_select = 30
rfe = RFE(estimator=logreg, n_features_to_select=num_features_to_select)

# Fit RFE to the scaled features
rfe.fit(data_scaled_df, y)

# Get the selected feature indices
selected_features_indices = rfe.get_support(indices=True)

# Filter the original feature names to retain only the selected features
selected_feature_names = data_scaled_df.columns[selected_features_indices]

# Transform scaled data with selected features back to DataFrame
selected_features = data_scaled_df.iloc[:, selected_features_indices]

# Create a DataFrame with only the selected features
selected_features_df = pd.DataFrame(selected_features, columns=selected_feature_names)

```

Checking RFE perforance


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Splitting the data 
X_train, X_test, y_train, y_test = train_test_split(selected_features_df, y, test_size=0.3, random_state=42)

logreg = LogisticRegression(solver='liblinear', random_state = 42, max_iter = 1000)
print("RFE Accuracy: {} ".format(logreg.fit(data_scaled_df, y).score(data_scaled_df, y)))
```

    RFE Accuracy: 0.9810126582278481 
    

Both methods perform well. In this project, we will opt for the RFE method.

#### Split the Data to Training, Test and Validation Set


```python
from sklearn.model_selection import train_test_split

# Our data is in X (features) and y (target)
# Splitting the data into 70% training and 30% test + validation combined
X_train, X_temp, y_train, y_temp = train_test_split(selected_features_df, y, test_size=0.3, random_state=42)

# Splitting the remaining 30% into equal parts for test (50%) and validation (50%)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Validation set size: {len(X_val)} samples")
```

    Training set size: 110 samples
    Test set size: 24 samples
    Validation set size: 24 samples
    

## Binary Classification

### Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
# Create a logistic regression model
logistic_regression = LogisticRegression()

# Define a grid of hyperparameters to search (example hyperparameters, adjust as needed)
param_grid = {
    'C': [0.1, 1.0, 10.0],  # Regularization parameter
    'solver': ['liblinear', 'lbfgs']  # Solver options
}
```

#### GridSearch with 10-Fold Cross Vaidation for optimization of Logistic Regression Model


```python
from sklearn.model_selection import GridSearchCV, cross_val_score
# Create a GridSearchCV object with the logistic regression model and parameter grid
grid_search = GridSearchCV(logistic_regression, param_grid, cv=10, scoring='accuracy')

# Fit the model on the training data to find the best hyperparameters
grid_search.fit(X_val, y_val)
```




    GridSearchCV(cv=10, estimator=LogisticRegression(),
                 param_grid={'C': [0.1, 1.0, 10.0],
                             'solver': ['liblinear', 'lbfgs']},
                 scoring='accuracy')




```python
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Train Best Parameters:", best_params)
print("Train Accuracy:", best_score)
```

    Train Best Parameters: {'C': 0.1, 'solver': 'liblinear'}
    Train Accuracy: 0.95
    


```python
best_model = grid_search.best_estimator_
print("Train model:", best_model)
```

    Train model: LogisticRegression(C=0.1, solver='liblinear')
    


```python
# Evaluate the best model using cross-validation
cv_scores = cross_val_score(best_model, X_test, y_test, cv=10)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
```

    Cross-validation scores: [0.66666667 0.33333333 1.         0.66666667 1.         0.5
     0.5        0.5        0.5        1.        ]
    Mean cross-validation score: 0.6666666666666666
    


```python
# Assuming cv_scores contains the cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
```


    
![png](README_files/README_59_0.png)
    



```python
mean_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

plt.figure(figsize=(8, 6))
plt.plot(range(len(mean_scores)), mean_scores, marker='o')
plt.xticks(range(len(mean_scores)), [str(p) for p in params], rotation=45)
plt.xlabel('Hyperparameters')
plt.ylabel('Mean Accuracy Score')
plt.title('Grid Search Results')
plt.tight_layout()
plt.show()
```


    
![png](README_files/README_60_0.png)
    


#### Evaluation of Logistic Regression Model using ROC-AUC and F1 Score  


```python
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
```


```python
# Predict probabilities for the validation set
y_probs = best_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC: {roc_auc:.4f}")
```

    ROC-AUC: 0.8671
    


```python
# Make predictions on the validation set
y_pred = best_model.predict(X_test)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")
```

    F1 Score: 0.7273
    


```python
# Make predictions on the validation set
y_pred_ = best_model.predict(X_val)

# Calculate Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
```

    Accuracy Score: 0.7500
    


```python

```


```python
compare_classification_methods.append({
        'Classification Method': "Logistic Regression",
        'Train Accuracy': best_score,
        'Test Accuracy': accuracy,
        'ROC-AUC Score': roc_auc,
        'F1 Score': f1
    })
```


```python
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
```


    
![png](README_files/README_68_0.png)
    


Logistic Regression model demonstrates a high accuracy during training (95%) but slightly lower accuracy during testing (75%). It shows a good overall performance but might be overfitting the data.

### K Nearest Neighbors

#### GridSearch with 10-Fold Cross Vaidation for optimization of K-Nearest Neighbors Model


```python
from sklearn.neighbors import KNeighborsClassifier

# Define the knn model
knn = KNeighborsClassifier()

# Define a grid of hyperparameters to search (e.g., different values of k)
param_grid = {'n_neighbors': np.arange(1, 15)}

# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_val, y_val)

# Get the best parameters
best_k = grid_search.best_params_['n_neighbors']

# Train the k-NN model using the best parameters
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Show the best hyperparameters and corresponding accuracy
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate Train Accuracy
train_accuracy = knn.fit(X_train, y_train).score(X_train, y_train)
print("Train accuracy: {} ".format(train_accuracy))
```

    Best hyperparameters: {'n_neighbors': 5}
    Train accuracy: 0.8727272727272727 
    


```python
# Extract grid search results
mean_scores = grid_search.cv_results_['mean_test_score']
k_values = np.arange(1, 15)

# Plot the grid search results
plt.figure(figsize=(8, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='-')
plt.title('Grid Search Results')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Accuracy')
plt.xticks(np.arange(1, 15, 2))
plt.grid(True)
plt.show()
```


    
![png](README_files/README_73_0.png)
    


#### Evaluation of K-Nearest Neighbors Model using ROC-AUC and F1 Score  


```python
# Predict probabilities for positive class
y_probs = best_knn.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_probs)

# Calculate F1 score
y_pred = best_knn.predict(X_test)
f1 = f1_score(y_test, y_pred)

# Tabulate Result
compare_classification_methods.append({
        'Classification Method': "K-Nearest Neighbors",
        'Train Accuracy': train_accuracy,
        'Test Accuracy': accuracy,
        'ROC-AUC Score': roc_auc,
        'F1 Score': f1
    })

# Get ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"ROC-AUC: {roc_auc}")
print(f"F1 Score: {f1}")


# Calculate Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

```


    
![png](README_files/README_75_0.png)
    


    ROC-AUC: 0.8391608391608391
    F1 Score: 0.7692307692307693
    Accuracy Score: 0.7500
    

K-Nearest Neighbors (KNN) model has a lower accuracy during training (87%) compared to other models, yet it shows similar accuracy during testing (75%). KNN might be simpler but lacks the robustness of other models.

### Support Vector Machines

#### GridSearch with 10-Fold Cross Vaidation for optimization of Support Vector Machines Model


```python
from sklearn.svm import SVC

# Define the SVM model
svm = SVC()

# Define the parameters grid for grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [10, 1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_val, y_val)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

# Calculate train accuracy using the best model
train_accuracy = best_svm.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)
```

    Best Parameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
    Training Accuracy: 1.0
    


```python
# Reshape results into a DataFrame
results = pd.DataFrame(grid_search.cv_results_)

# Plotting mean test scores against hyperparameters
for kernel in param_grid['kernel']:
    kernel_results = results[results['param_kernel'] == kernel]

    plt.figure(figsize=(8, 6))
    for i, val in enumerate(param_grid['C']):
        plt.plot(
            kernel_results[kernel_results['param_C'] == val]['param_gamma'],
            kernel_results[kernel_results['param_C'] == val]['mean_test_score'],
            marker='o',
            label=f'C: {val}'
        )

    plt.title(f'Mean Test Scores for {kernel} Kernel')
    plt.xlabel('Gamma')
    plt.ylabel('Mean Test Score')
    plt.legend(title='C')
    plt.show()
```


    
![png](README_files/README_80_0.png)
    



    
![png](README_files/README_80_1.png)
    


#### Evaluation of Support Vector Machines Model using ROC-AUC and F1 Score  


```python
# Get predicted probabilities for the positive class
y_pred_prob = best_svm.decision_function(X_test)

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# F1 score
y_pred = best_svm.predict(X_test)
f1 = f1_score(y_test, y_pred)

# Calculate Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Tabulate Result
compare_classification_methods.append({
        'Classification Method': "Support Vector Machines",
        'Train Accuracy': train_accuracy,
        'Test Accuracy': accuracy,
        'ROC-AUC Score': roc_auc,
        'F1 Score': f1
    })

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.show()

print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")


```

    Accuracy Score: 0.8750
    


    
![png](README_files/README_82_1.png)
    


    ROC AUC: 0.8741
    F1 Score: 0.8800
    

Support Vector Machines (SVM) model have a perfect accuracy during training (100%) and good accuracy during testing (87.5%). This indicates a strong ability to learn patterns in the data and generalize well to new, unseen data.

### Random Forest

#### GridSearch with 10-Fold Cross Vaidation for optimization of Random Forest Model


```python
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Define the hyperparameters grid for Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with 10-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_val, y_val)

# Get the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

# Train the model using the best parameters
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)

# Calculate the training accuracy
train_accuracy = best_rf.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)
```

    Best Parameters: {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 50}
    Training Accuracy: 1.0
    


```python
# Extract scores and parameters from the grid search results
scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Get the parameter combinations tested
param_combinations = [str(param) for param in params]

# Get the best parameters
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Plotting the scores for different parameter combinations
plt.figure(figsize=(13, 15))
plt.barh(param_combinations, scores, color='skyblue')

# Highlight the bar for the best parameter combination
for i, param in enumerate(param_combinations):
    if params[i] == best_params:
        plt.barh(param, scores[i], color='orange')

plt.xlabel('Mean Test Score')
plt.title('Random Forest Grid Search Results')
plt.axvline(x=best_score, color='red', linestyle='--', label='Best Score: {:.2f}'.format(best_score))
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](README_files/README_87_0.png)
    


#### Evaluation of Random Forest Model using ROC-AUC and F1 Score  


```python
from sklearn.metrics import confusion_matrix
# Calculate ROC-AUC score
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC Score:", roc_auc)

# Calculate F1 score
y_pred = best_rf.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# Calculate Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Tabulate Result
compare_classification_methods.append({
        'Classification Method': "Random Forest",
        'Train Accuracy': train_accuracy,
        'Test Accuracy': accuracy,
        'ROC-AUC Score': roc_auc,
        'F1 Score': f1
    })

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
```

    ROC AUC Score: 0.881118881118881
    F1 Score: 0.9230769230769231
    Accuracy: 0.9166666666666666
    


    
![png](README_files/README_89_1.png)
    



    
![png](README_files/README_89_2.png)
    


The model display perfect accuracy during training (100%) and high accuracy during testing (91.67%). These models show promising results, indicating strong predictive capabilities and generalization to new data.

### Extreme Gradient Boosting


```python
# Instaling XGBoost package library
!pip install xgboost
```

    Requirement already satisfied: xgboost in c:\programdata\anaconda3\lib\site-packages (1.6.2)
    Requirement already satisfied: scipy in c:\users\mneme\appdata\roaming\python\python37\site-packages (from xgboost) (1.7.3)
    Requirement already satisfied: numpy in c:\users\mneme\appdata\roaming\python\python37\site-packages (from xgboost) (1.16.5)
    

#### GridSearch with 10-Fold Cross Vaidation for optimization of Extreme Gradient Boosting Model


```python
import xgboost as xgb

# Define the parameter grid for grid search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300]
}

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Perform grid search with 10-fold cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_val, y_val)

# Show best parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_xgb = xgb.XGBClassifier(**grid_search.best_params_)
best_xgb.fit(X_train, y_train)

# Get the training accuracy
train_accuracy = best_xgb.score(X_train, y_train)
print("Training accuracy: {:.2f}%".format(train_accuracy * 100))

# You can also get cross-validation scores
cross_val_scores = cross_val_score(best_xgb, X_train, y_train, cv=10)
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation accuracy: {:.2f}%".format(cross_val_scores.mean() * 100))
```

    Best parameters found:  {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
    Training accuracy: 100.00%
    Cross-validation scores: [1.         0.63636364 0.90909091 1.         1.         0.90909091
     0.81818182 1.         0.81818182 0.90909091]
    Mean cross-validation accuracy: 90.00%
    

#### Evaluation of Extreme Gradient Boosting Model using ROC-AUC and F1 Score  


```python
# Get the predicted probabilities for the positive class
y_pred = best_xgb.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate F1 score
y_pred_binary = best_xgb.predict(X_test)
f1 = f1_score(y_test, y_pred_binary)

print("ROC-AUC:", roc_auc)
print("F1 Score:", f1)

# Calculate Accuracy score
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

# Tabulate Result
compare_classification_methods.append({
        'Classification Method': "Extreme Gradient Boosting",
        'Train Accuracy': train_accuracy,
        'Test Accuracy': accuracy,
        'ROC-AUC Score': roc_auc,
        'F1 Score': f1
    })
```

    ROC-AUC: 0.8951048951048951
    F1 Score: 0.9230769230769231
    Accuracy: 0.9166666666666666
    


```python
fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```


    
![png](README_files/README_97_0.png)
    


The model display perfect accuracy during training (100%) and high accuracy during testing (91.67%). These models show promising results, indicating strong predictive capabilities and generalization to new data.

## Summary


```python
# Result Tabulation 
comparison = pd.DataFrame(compare_classification_methods)
comparison.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classification Method</th>
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>ROC-AUC Score</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Logistic Regression</td>
      <td>0.950000</td>
      <td>0.750000</td>
      <td>0.867133</td>
      <td>0.727273</td>
    </tr>
    <tr>
      <td>1</td>
      <td>K-Nearest Neighbors</td>
      <td>0.872727</td>
      <td>0.750000</td>
      <td>0.839161</td>
      <td>0.769231</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Support Vector Machines</td>
      <td>1.000000</td>
      <td>0.875000</td>
      <td>0.874126</td>
      <td>0.880000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Random Forest</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>0.881119</td>
      <td>0.923077</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Extreme Gradient Boosting</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>0.895105</td>
      <td>0.923077</td>
    </tr>
  </tbody>
</table>
</div>



In summary, among the models, Support Vector Machines, Random Forest, and Extreme Gradient Boosting (XGBoost) stand out. They all achieved high training accuracy, good test accuracy, and strong ROC-AUC and F1 scores. Logistic Regression and K-Nearest Neighbors have slightly lower performance, with decent test accuracy but less robust ROC-AUC and F1 scores. The choice of the best method depends on your specific goals and the trade-offs you are willing to make between training and testing performance.


```python

```
