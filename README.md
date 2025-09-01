# Replication of results from "Prediction of ceftriaxone MIC in <i>Neisseria gonorrhoeae</i> using DNA microarray technology and regression analysis" in Python

&emsp;In the original paper "Prediction of ceftriaxone MIC in <i>Neisseria gonorrhoeae</i> using DNA microarray technology and regression analysis" (https://doi.org/10.1093/jac/dkab308) minimal inhibitory concentration (MIC) prediction is carried out in R. I set myself a goal to replicate the results using Python. The main objective of the work is to train a linear regression model that would be capable of prediciting MIC of ceftriaxone based on mutations present in N. gonorrhoeae genome.

## Data analysis and preprocessing


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso
```

&emsp;Loading the data with pandas:


```python
df = pd.read_excel("./Table_S1.xlsx", header=3, index_col=0)
df.head()
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
      <th>penA allele</th>
      <th>Name</th>
      <th>Date</th>
      <th>Collection</th>
      <th>Country</th>
      <th>Ceftriaxone MIC, mg/L</th>
      <th>Ala311Val</th>
      <th>Ile312Met</th>
      <th>Val316Thr</th>
      <th>insAsp(345-346)</th>
      <th>...</th>
      <th>Gly120Lys</th>
      <th>Gly120Arg</th>
      <th>Ala121Gly</th>
      <th>Ala121Ser</th>
      <th>Ala121Asp</th>
      <th>Ala121Asn</th>
      <th>Ala121Val</th>
      <th>Ala121Arg</th>
      <th>Leu421Pro</th>
      <th>-35delA</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>100.001</td>
      <td>SRR1661159</td>
      <td>2001</td>
      <td>demczuk2015</td>
      <td>Canada</td>
      <td>0.00025</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100.001</td>
      <td>SRR1661178</td>
      <td>2005</td>
      <td>demczuk2015</td>
      <td>Canada</td>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100.001</td>
      <td>SRR1661264</td>
      <td>2010</td>
      <td>demczuk2015</td>
      <td>Canada</td>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.001</td>
      <td>SRR1661260</td>
      <td>2010</td>
      <td>demczuk2015</td>
      <td>Canada</td>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15.001</td>
      <td>ECDC_T2_ES012</td>
      <td>2013</td>
      <td>eurogasp2013</td>
      <td>Estonia</td>
      <td>0.00070</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5812 entries, 1 to 5812
    Data columns (total 31 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   penA allele            5812 non-null   float64
     1   Name                   5812 non-null   object 
     2   Date                   5812 non-null   object 
     3   Collection             5812 non-null   object 
     4   Country                5812 non-null   object 
     5   Ceftriaxone MIC, mg/L  5812 non-null   float64
     6   Ala311Val              5812 non-null   int64  
     7   Ile312Met              5812 non-null   int64  
     8   Val316Thr              5812 non-null   int64  
     9   insAsp(345-346)        5812 non-null   int64  
     10  Thr483Ser              5812 non-null   int64  
     11  Ala501Val              5812 non-null   int64  
     12  Ala501Thr              5812 non-null   int64  
     13  Ala501Pro              5812 non-null   int64  
     14  Asn512Tyr              5812 non-null   int64  
     15  Gly542Ser              5812 non-null   int64  
     16  Gly545Ser              5812 non-null   int64  
     17  Pro551Leu              5812 non-null   int64  
     18  Pro551Ser              5812 non-null   int64  
     19  Gly120Asn              5812 non-null   int64  
     20  Gly120Asp              5812 non-null   int64  
     21  Gly120Lys              5812 non-null   int64  
     22  Gly120Arg              5812 non-null   int64  
     23  Ala121Gly              5812 non-null   int64  
     24  Ala121Ser              5812 non-null   int64  
     25  Ala121Asp              5812 non-null   int64  
     26  Ala121Asn              5812 non-null   int64  
     27  Ala121Val              5812 non-null   int64  
     28  Ala121Arg              5812 non-null   int64  
     29  Leu421Pro              5812 non-null   int64  
     30  -35delA                5812 non-null   int64  
    dtypes: float64(2), int64(25), object(4)
    memory usage: 1.4+ MB
    

The DataFrame contained 5812 rows and 31 columns:<br>
- penA allele - type of penA allele designated by a number (categorical variable);<br>
- Name - SRA Run Id;<br>
- Date - the year when the data was uploaded;<br>
- Collection - name of the collection from which the data comes from;<br>
- Country - origin of the isolate;<br>
- Ceftriaxone MIC, mg/L - minimal inhibitory concentration of ceftriaxone in mg/L (float);<br>
- Ala311Val...-35delA - mutations in N. gonorrhoeae genome that are responsible for resistance (boolean).<br>


```python
# Delete duplicate rows
df = df.drop_duplicates()
```

Among 5812 rows there were 95 duplicates. These were definitely duplicates as they shared the same SRA Run Id. The remaining amount of rows is 5717.


```python
cm = 1 / 2.54  # cm to inches conversion
fig, ax = plt.subplots(1, 1, figsize=(16 * cm, 10 * cm))
mnso.matrix(df, ax=ax, fontsize=8, sparkline=False)
plt.title("Matrix of missing values")
```




    Text(0.5, 1.0, 'Matrix of missing values')




    
![png](README_files/README_10_1.png)
    


As we can see, the dataset contains no missing values.

To train a model we need features (columns: Ala311Val...-35delA) and a target variable (column: Ceftriaxone MIC, mg/L). The other columns were not used in the analysis in the paper.


```python
# Dropping columns that were not used in the analysis.
df = df.drop(axis=1, columns=df.columns[range(5)])

# Renaming target variable for the ease of use.
df = df.rename({"Ceftriaxone MIC, mg/L": "cft_mic"}, axis=1)
df.head()
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
      <th>cft_mic</th>
      <th>Ala311Val</th>
      <th>Ile312Met</th>
      <th>Val316Thr</th>
      <th>insAsp(345-346)</th>
      <th>Thr483Ser</th>
      <th>Ala501Val</th>
      <th>Ala501Thr</th>
      <th>Ala501Pro</th>
      <th>Asn512Tyr</th>
      <th>...</th>
      <th>Gly120Lys</th>
      <th>Gly120Arg</th>
      <th>Ala121Gly</th>
      <th>Ala121Ser</th>
      <th>Ala121Asp</th>
      <th>Ala121Asn</th>
      <th>Ala121Val</th>
      <th>Ala121Arg</th>
      <th>Leu421Pro</th>
      <th>-35delA</th>
    </tr>
    <tr>
      <th>#</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00025</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00050</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.00070</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Changing variable type (less memory usage).
df.iloc[:, 1:] = df.iloc[:, 1:].astype("Int8")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5717 entries, 1 to 5812
    Data columns (total 26 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   cft_mic          5717 non-null   float64
     1   Ala311Val        5717 non-null   int64  
     2   Ile312Met        5717 non-null   int64  
     3   Val316Thr        5717 non-null   int64  
     4   insAsp(345-346)  5717 non-null   int64  
     5   Thr483Ser        5717 non-null   int64  
     6   Ala501Val        5717 non-null   int64  
     7   Ala501Thr        5717 non-null   int64  
     8   Ala501Pro        5717 non-null   int64  
     9   Asn512Tyr        5717 non-null   int64  
     10  Gly542Ser        5717 non-null   int64  
     11  Gly545Ser        5717 non-null   int64  
     12  Pro551Leu        5717 non-null   int64  
     13  Pro551Ser        5717 non-null   int64  
     14  Gly120Asn        5717 non-null   int64  
     15  Gly120Asp        5717 non-null   int64  
     16  Gly120Lys        5717 non-null   int64  
     17  Gly120Arg        5717 non-null   int64  
     18  Ala121Gly        5717 non-null   int64  
     19  Ala121Ser        5717 non-null   int64  
     20  Ala121Asp        5717 non-null   int64  
     21  Ala121Asn        5717 non-null   int64  
     22  Ala121Val        5717 non-null   int64  
     23  Ala121Arg        5717 non-null   int64  
     24  Leu421Pro        5717 non-null   int64  
     25  -35delA          5717 non-null   int64  
    dtypes: float64(1), int64(25)
    memory usage: 1.2 MB
    


```python
df = df.reset_index(drop=True)  # Reset index (duplicate columns were dropped).
```


```python
y = df.iloc[:, 0]  # Target variable.

X = df.iloc[:, 1:]  # Features.
```

It is a good idea to analyse target variable distribution before training a model.


```python
sns.histplot(y)
plt.title("Ceftriaxone MIC distribution")
plt.xlabel("Ceftriaxone MIC, mg/L")
plt.vlines(x=[0.125], ymin=0, ymax=4000, colors="black", linestyles="dashed")
plt.annotate(
    "",
    xytext=(0.500, 2000),
    xy=(0.125, 2000),
    arrowprops=dict(arrowstyle="->"),
    horizontalalignment="center",
    verticalalignment="bottom",
)
plt.text(s="MIC breakpoint", x=0.17, y=2100)
plt.ylim((0, 3000))
plt.show()
```


    
![png](README_files/README_19_0.png)
    


As we can see most of the observations lie before the MIC breakpoint (black dashed line at 0.125 mg/L). A few data points of resistant isolates are found after the breakpoint. It is certain that the distribution is not normal (right-skewed) and for that reason the target variable should be transformed (for this case log transformation would be appropriate).


```python
sns.histplot(np.log2(y))
plt.title("Ceftriaxone MIC distribution")
plt.xlabel("log2(Ceftriaxone MIC)")
plt.show()
```


    
![png](README_files/README_21_0.png)
    


Now the data looks more bell-shaped. Now linear regression can be used on this data.

## Training linear regression model


```python
from sklearn.model_selection import train_test_split

# Splitting dataset into training (75%) and test (25%) data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=8798
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (4287, 25) (1430, 25) (4287,) (1430,)
    


```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer

# Exctract feature names.
features = X.columns

# Impute missing values in case they will be encountered during prediction.
preprocessor = ColumnTransformer(
    transformers=[("imputer", SimpleImputer(strategy="most_frequent"), features)]
)

# Transform target variable with log2.
target_transformation_regression = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=lambda x: np.log2(x),
    inverse_func=lambda x: np.exp2(x),
)

# Creating a pipeline with preprocessor and regression steps.
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("linear_regression", target_transformation_regression),
    ]
)
```


```python
# Fit data.
linear_model.fit(X_train, y_train)
print()
```

    
    


```python
from sklearn.metrics import r2_score, mean_squared_error


def print_statistics(y_true_train, X_train, y_true_test, X_test, model):
    """
    A function to calculate R² and MSE for train and test data.
    """
    print(f"Train statistics: ")
    print("R² = {:.4f}".format(r2_score(y_true_train, model.predict(X_train))))
    print(
        "MSE = {:.4e}".format(mean_squared_error(y_true_train, model.predict(X_train)))
    )
    print()
    print(f"Test statistics: ")
    print("R² = {:.4f}".format(r2_score(y_true_test, model.predict(X_test))))
    print("MSE = {:.4e}".format(mean_squared_error(y_true_test, model.predict(X_test))))
    print()
```


```python
# Statistics of the trained model.
print_statistics(y_train, X_train, y_test, X_test, linear_model)
```

    Train statistics: 
    R² = 0.8462
    MSE = 5.8080e-04
    
    Test statistics: 
    R² = 0.7059
    MSE = 3.0906e-04
    
    

From calculated statistics we can see that a linear regression with log2 transformation is not ideal but it gives moderately good resulsts. Let's see the coefficients of the model:


```python
coefficients = pd.DataFrame(
    linear_model[1].regressor_.coef_, columns=["Coefficients"], index=features
)
coefficients.sort_values(by="Coefficients", key=lambda x: abs(x), ascending=False)
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
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ala501Pro</th>
      <td>5.211000</td>
    </tr>
    <tr>
      <th>Gly120Arg</th>
      <td>2.117484</td>
    </tr>
    <tr>
      <th>Ala311Val</th>
      <td>1.662172</td>
    </tr>
    <tr>
      <th>Thr483Ser</th>
      <td>1.662172</td>
    </tr>
    <tr>
      <th>Gly545Ser</th>
      <td>1.656373</td>
    </tr>
    <tr>
      <th>insAsp(345-346)</th>
      <td>1.333320</td>
    </tr>
    <tr>
      <th>Gly542Ser</th>
      <td>1.242723</td>
    </tr>
    <tr>
      <th>Ala501Val</th>
      <td>1.226636</td>
    </tr>
    <tr>
      <th>Gly120Lys</th>
      <td>1.204336</td>
    </tr>
    <tr>
      <th>Pro551Ser</th>
      <td>1.021032</td>
    </tr>
    <tr>
      <th>Ala121Val</th>
      <td>-0.959558</td>
    </tr>
    <tr>
      <th>Pro551Leu</th>
      <td>0.744049</td>
    </tr>
    <tr>
      <th>Ala501Thr</th>
      <td>0.710691</td>
    </tr>
    <tr>
      <th>Asn512Tyr</th>
      <td>0.670171</td>
    </tr>
    <tr>
      <th>Gly120Asn</th>
      <td>0.610457</td>
    </tr>
    <tr>
      <th>Val316Thr</th>
      <td>0.531650</td>
    </tr>
    <tr>
      <th>Ile312Met</th>
      <td>0.531650</td>
    </tr>
    <tr>
      <th>Gly120Asp</th>
      <td>0.500875</td>
    </tr>
    <tr>
      <th>Ala121Arg</th>
      <td>0.439970</td>
    </tr>
    <tr>
      <th>Ala121Gly</th>
      <td>-0.274222</td>
    </tr>
    <tr>
      <th>Ala121Ser</th>
      <td>0.239992</td>
    </tr>
    <tr>
      <th>Leu421Pro</th>
      <td>0.139063</td>
    </tr>
    <tr>
      <th>Ala121Asp</th>
      <td>-0.074335</td>
    </tr>
    <tr>
      <th>Ala121Asn</th>
      <td>0.039008</td>
    </tr>
    <tr>
      <th>-35delA</th>
      <td>-0.033247</td>
    </tr>
  </tbody>
</table>
</div>



It is better to visualise our coefficients using horizontal barplot:


```python
coefficients.sort_values(by="Coefficients").plot.barh(
    legend=False,
    xlabel="Coefficient value, mg/L",
    ylabel="Coefficient name",
    title="Horizontal barplot of linear regression coefficients",
)
plt.grid(alpha=0.5)
plt.show()
```


    
![png](README_files/README_32_0.png)
    


Horizontal barplot shows clearly that there are variables that strongly affect MIC of ceftriaxone like: Ala501Pro, Gly120Arg, Thr483Ser, Gly545Ser, insAsp(345-346), Gly542Ser, Ala501Val, Gly120Lys, Pro551Ser, Ala121Val. There are also variables that have almost no effect on the target variable like: Ala121Asp, -35delA, Ala121Asn, Leu421Pro, Ala121Ser, Ala121Gly. Other variables moderately affect the target variable.

We can try to remove variables that have no effect on the target variable and retrain the model.

## Retrain model with less variables


```python
features_reduced = [
    i
    for i in X.columns
    if i
    not in ["Ala121Asp", "-35delA", "Ala121Asn", "Leu421Pro", "Ala121Ser", "Ala121Gly"]
]

# Impute missing values in case they will be encountered during prediction.
preprocessor_reduced = ColumnTransformer(
    transformers=[
        ("imputer", SimpleImputer(strategy="most_frequent"), features_reduced)
    ]
)

# Creating a pipeline with preprocessor and regression steps.
linear_model_reduced = Pipeline(
    steps=[
        ("preprocessor", preprocessor_reduced),
        ("linear_regression", target_transformation_regression),
    ]
)
```


```python
linear_model_reduced.fit(X_train[features_reduced], y_train)
print()
```

    
    


```python
print_statistics(
    y_train,
    X_train[features_reduced],
    y_test,
    X_test[features_reduced],
    linear_model_reduced,
)
```

    Train statistics: 
    R² = 0.8461
    MSE = 5.8110e-04
    
    Test statistics: 
    R² = 0.7048
    MSE = 3.1019e-04
    
    

Let's compare the results of the original and the new model with reduced amount of variables:<br>
1) Train statistics:<br>
> Origianl: R² = 0.8462 | New: R² = 0.8461;<br>
> Origianl: MSE = 5.8080e-04 | New: MSE = 5.8110e-04.<br>
Train statistics remained largely the same.<br>

2) Test statistics:<br>
> Origianl: R² = 0.7059 | New: R² = 0.7048;<br>
> Origianl: MSE = 3.0906e-04 | New: MSE = 3.1019e-04.<br>
Test statistics remained largely the same.<br>

We can conclude that removed variables were not important for the linear regression model.

## The model proposed in the paper

In the original paper the Akaike information criterion was used to determine unimportant variables. Features that were removed are: Val316Thr, Thr483Ser, Ala121Asn, Ala121Arg, -35delA (only -35delA and Ala121Asn were both removed from my model and from the author's model).<br>
Now I will train and test the proposed model and compare it to the model that I previously trained.


```python
features_paper = [
    i
    for i in X.columns
    if i not in ["Val316Thr", "Thr483Ser", "Ala121Asn", "Ala121Arg", "-35delA"]
]

# Impute missing values in case they will be encountered during prediction.
preprocessor_paper = ColumnTransformer(
    transformers=[("imputer", SimpleImputer(strategy="most_frequent"), features_paper)]
)

# Creating a pipeline with preprocessor and regression steps.
linear_model_paper = Pipeline(
    steps=[
        ("preprocessor", preprocessor_paper),
        ("linear_regression", target_transformation_regression),
    ]
)
```


```python
linear_model_paper.fit(X_train[features_paper], y_train)
print()
```

    
    


```python
print_statistics(
    y_train, X_train[features_paper], y_test, X_test[features_paper], linear_model_paper
)
```

    Train statistics: 
    R² = 0.8461
    MSE = 5.8098e-04
    
    Test statistics: 
    R² = 0.7060
    MSE = 3.0896e-04
    
    


```python
coefficients_paper = pd.DataFrame(
    linear_model_paper[1].regressor_.coef_,
    columns=["Coefficients"],
    index=features_paper,
)
coefficients_paper.sort_values(by="Coefficients", key=lambda x: abs(x), ascending=False)
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
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ala501Pro</th>
      <td>5.209523</td>
    </tr>
    <tr>
      <th>Ala311Val</th>
      <td>3.322759</td>
    </tr>
    <tr>
      <th>Gly120Arg</th>
      <td>2.446465</td>
    </tr>
    <tr>
      <th>Gly545Ser</th>
      <td>1.654285</td>
    </tr>
    <tr>
      <th>insAsp(345-346)</th>
      <td>1.333082</td>
    </tr>
    <tr>
      <th>Gly542Ser</th>
      <td>1.243511</td>
    </tr>
    <tr>
      <th>Gly120Lys</th>
      <td>1.234166</td>
    </tr>
    <tr>
      <th>Ala501Val</th>
      <td>1.225806</td>
    </tr>
    <tr>
      <th>Ile312Met</th>
      <td>1.061330</td>
    </tr>
    <tr>
      <th>Pro551Ser</th>
      <td>1.017283</td>
    </tr>
    <tr>
      <th>Ala121Val</th>
      <td>-0.982810</td>
    </tr>
    <tr>
      <th>Pro551Leu</th>
      <td>0.743698</td>
    </tr>
    <tr>
      <th>Ala501Thr</th>
      <td>0.706470</td>
    </tr>
    <tr>
      <th>Asn512Tyr</th>
      <td>0.670467</td>
    </tr>
    <tr>
      <th>Gly120Asn</th>
      <td>0.633672</td>
    </tr>
    <tr>
      <th>Gly120Asp</th>
      <td>0.510602</td>
    </tr>
    <tr>
      <th>Ala121Gly</th>
      <td>-0.281977</td>
    </tr>
    <tr>
      <th>Ala121Ser</th>
      <td>0.238801</td>
    </tr>
    <tr>
      <th>Leu421Pro</th>
      <td>0.119956</td>
    </tr>
    <tr>
      <th>Ala121Asp</th>
      <td>-0.113236</td>
    </tr>
  </tbody>
</table>
</div>



The coefficients obtained are almost identical to the ones calculated in the article (dew to the nature of randomness in training).

Let's compare the results of the original and the new model with reduced amount of variables:<br>
1) Train statistics:<br>
> Paper model: R² = 0.8461 | My model: R² = 0.8461;<br>
> Paper model: MSE = 5.8098e-04 | My model: MSE = 5.8110e-04.<br>
Train statistics is almost the same.<br>

2) Test statistics:<br>
> Paper model: R² = 0.7060 | My model: R² = 0.7048;<br>
> Paper model: MSE = 3.0896e-04 | My model: MSE = 3.1019e-04.<br>
Test statistics is almost the same.<br>

We can see that the models perfrom equally well with almost identical R² coefficients and MSE. This tells us that linear regression model might not be the optimal one since even with differen variables present the result stays largely the same.

## Conclusion

1) In this work a dataset from https://doi.org/10.1093/jac/dkab308 was analysed. There were found 95 duplicate rows in the dataset which reduced the amount of observations from 5812 to 5717. No missing values were found.<br>
2) The target variable in the dataset was non-normally distributed (right-skewed). Log2 transformation was used to make the distribution look more bell-shaped.<br>
3) The first trained model with no features deleted performed pretty well with moderately high R² coefficient and low MSE on training and test data (Train statistics: R² = 0.8462, MSE = 5.8080e-04 | Test statistics: R² = 0.7059, MSE = 3.0906e-04).<br>
4) The model with reduced amount of features performed as good as the first model (Train statistics: R² = 0.8461, MSE = 5.8110e-04 | Test statistics: R² = 0.7048, MSE = 3.1019e-04).<br>
5) The model proposed in the paper performed as well as my models (Train statistics: R² = 0.8461,MSE = 5.8098e-04 | Test statistics: R² = 0.7060, MSE = 3.1019e-04).<br>
6) The proposed model coefficients were almost identical to the ones calculated in the article (dew to the nature of randomness in the training process).<br>
7) It could be speculated that the linear regression model is not ideal for this type of problem since removing different coefficients results in largely the same model behaviour. On top of that not all of the genetic determinants of resistance are found leading to an acceptable R² coefficient value (<0.9).

