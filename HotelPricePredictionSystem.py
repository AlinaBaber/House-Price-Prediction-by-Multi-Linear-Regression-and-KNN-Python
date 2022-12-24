import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from scipy.stats import skew
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats  # For in-built method to get PCC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression
#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
#Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

#config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
#matplotlib inline
class House_Price_Prediction_System():
    def __init__(self, train_path="Train.xlsx", test_path="Test.xlsx"):
        self.Train = train_path
        self.Test = test_path



    def data_acquisition(self):
        # Data Acquisition
        train = pd.read_excel("Train.xlsx")
        test = pd.read_excel("Test.xlsx")
        # Save the 'Id' column
        train_ID = train['Id']
        test_ID = test['Id']

        # Now drop the 'Id' column since it's unnecessary for the prediction process.
        train.drop("Id", axis=1, inplace=True)
        test.drop("Id", axis=1, inplace=True)
        # Check the numbers of samples and features
        print("The train data size before dropping Id feature is : {} ".format(train.shape))
        print("The test data size before dropping Id feature is : {} ".format(test.shape))
        print(train.head())
        print(test.head())

        self.train = train
        self.test= train
        self.train_ID = train_ID
        self.test_ID = test_ID

    def missing_values_treatment(self):
        train = self.train
        # check missing values
        train.columns[train.isnull().any()]
        # missing value counts in each of these columns
        miss = train.isnull().sum() / len(train)
        miss = miss[miss > 0]
        miss.sort_values(inplace=True)
        print("Missing values", miss)
        # visualising missing values
        miss = miss.to_frame()
        miss.columns = ['count']
        miss.index.names = ['Name']
        miss['Name'] = miss.index

        # plot the missing value count
        sns.set(style="whitegrid", color_codes=True)
        sns.barplot(x='Name', y='count', data=miss)
        plt.xticks(rotation=90)
        plt.title('With Missing value Data Histogram')
        plt.show()
        # Plot Histogram
        sns.distplot(train['SalePrice'], fit=norm);

        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(train['SalePrice'])
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('With Missing value SalePrice distribution')

        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        plt.show()

    def saleprice_variable_statistical_analysis(self):
        train = self.train
        train_des=train['SalePrice'].describe()
        print(train_des)
        # Plot Histogram
        sns.distplot(train['SalePrice'], fit=norm);

        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(train['SalePrice'])
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        plt.show()

        print("Skewness: %f" % train['SalePrice'].skew())
        print("Kurtosis: %f" % train['SalePrice'].kurt())
        self.mu =mu
        self.sigma = sigma

    def correlation_matrix(self):
        train = self.train
        # separate variables into new data frames
        numeric_data = train.select_dtypes(include=[np.number])
        cat_data = train.select_dtypes(exclude=[np.number])
        print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],
                                                                                     cat_data.shape[1]))
       # del numeric_data['id']
        # correlation plot
        corr = numeric_data.corr()
        sns.heatmap(corr)

    def multivariable_feature_analysis(self):
        train = self.train
        # Checking Categorical Data
        train.select_dtypes(include=['object'])
        #print(train.columns.str.isnumeric())
       # for num in train.columns.str.isnumeric():
       #     sns.boxplot(x=train[num], y=train['SalePrice'])
        #    plt.show()
        # Checking Numerical Data
        train.select_dtypes(include=['int64', 'float64']).columns
        cat = len(train.select_dtypes(include=['object']).columns)
        num = len(train.select_dtypes(include=['int64', 'float64']).columns)
        print('Total Features: ', cat, 'categorical', '+',
              num, 'numerical', '=', cat + num, 'features')
        # Correlation Matrix Heatmap
        corrmat = train.corr()
        print("Correlation Matrix",corrmat)
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        plt.title('Correlation Matrix')
        plt.show()
        # Top 10 Heatmap
        k = 10  # number of variables for heatmap
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        print("cols", cols)
        cm = np.corrcoef(train[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.title('10 Variables Correlation Matrix')
        plt.show()
        most_corr = pd.DataFrame(cols)
        most_corr.columns = ['Most Correlated Features']
        print(most_corr)
        corrs = most_corr.values
        trains = dict()
        #for i in corrs:
        #    print(i, type(str(i)))
        #    s=str(i)
        #    a=train[s].
        #    trains.update({s: a})
        #print(trains.values())
        # Overall Quality vs Sale Price
        var = 'OverallQual'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)
        plt.show()
        # Living Area vs Sale Price
        sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Removing outliers manually (Two points in the bottom right)
        train = train.drop(train[(train['GrLivArea'] > 4000)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Living Area vs Sale Price
        sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Garage Area vs Sale Price
        sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
        plt.show()
        # Removing outliers manually (More than 4-cars, less than $300k)
        train = train.drop(train[(train['GarageCars'] > 3)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Garage Area vs Sale Price
        sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
        plt.show()
        # Garage Area vs Sale Price
        sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Removing outliers manually (More than 1000 sqft, less than $300k)
        train = train.drop(train[(train['GarageArea'] > 1000)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Garage Area vs Sale Price
        sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Basement Area vs Sale Price
        sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')
        plt.show()
        # First Floor Area vs Sale Price
        sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg')
        plt.show()
        # FullBath vs Sale Price
        var = 'FullBath'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)
        plt.show()
        # Total Rooms vs Sale Price
        sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'])
        plt.show()
        # Total Rooms vs Sale Price
        var = 'YearBuilt'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(16, 8))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000);
        plt.xticks(rotation=90);
        plt.show()
        self.train = train
        #self.data = data
        self.cols = cols

    def treat_missing_values_outlier(self):
        # Combining Datasets
        train = self.train
        test=self.test
        ntrain = train.shape[0]
        ntest = test.shape[0]
        self.ntrain =ntrain
        self.ntest = ntest
        y_train = train.SalePrice.values
        all_data = pd.concat((train, test)).reset_index(drop=True)
        all_data.drop(['SalePrice'], axis=1, inplace=True)

        print("Train data size is : {}".format(train.shape))
        print("Test data size is : {}".format(test.shape))
        print("Combined dataset size is : {}".format(all_data.shape))
        # Find Missing Ratio of Dataset
        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(missing_data)
        # Percent missing data by feature
        f, ax = plt.subplots(figsize=(15, 12))
        plt.xticks(rotation='90')
        sns.barplot(x=all_data_na.index, y=all_data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
        plt.show()
        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
        all_data["Alley"] = all_data["Alley"].fillna("None")
        all_data["Fence"] = all_data["Fence"].fillna("None")
        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            all_data[col] = all_data[col].fillna('None')
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            all_data[col] = all_data[col].fillna(0)
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            all_data[col] = all_data[col].fillna(0)
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            all_data[col] = all_data[col].fillna('None')
        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
        all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
        all_data = all_data.drop(['Utilities'], axis=1)
        all_data["Functional"] = all_data["Functional"].fillna("Typ")
        all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
        all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
        all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
        all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
        all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

        # Check if there are any missing values left
        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        missing_data.head()
        print(all_data_na)
        print(missing_data)
        self.all_data = all_data
        self.train =train
        self.test =test
        # Overall Quality vs Sale Price
        var = 'OverallQual'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)
        plt.show()
        # Living Area vs Sale Price
        sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Removing outliers manually (Two points in the bottom right)
        train = train.drop(train[(train['GrLivArea'] > 4000)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Living Area vs Sale Price
        sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Garage Area vs Sale Price
        sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
        plt.show()
        # Removing outliers manually (More than 4-cars, less than $300k)
        train = train.drop(train[(train['GarageCars'] > 3)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Garage Area vs Sale Price
        sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
        plt.show()
        # Garage Area vs Sale Price
        sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Removing outliers manually (More than 1000 sqft, less than $300k)
        train = train.drop(train[(train['GarageArea'] > 1000)
                                 & (train['SalePrice'] < 300000)].index).reset_index(drop=True)
        # Garage Area vs Sale Price
        sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
        plt.show()
        # Basement Area vs Sale Price
        sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')
        plt.show()
        # First Floor Area vs Sale Price
        sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg')
        plt.show()
        # FullBath vs Sale Price
        var = 'FullBath'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(8, 6))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000)
        plt.show()
        # Total Rooms vs Sale Price
        sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'])
        plt.show()
        # Total Rooms vs Sale Price
        var = 'YearBuilt'
        data = pd.concat([train['SalePrice'], train[var]], axis=1)
        f, ax = plt.subplots(figsize=(16, 8))
        fig = sns.boxplot(x=var, y="SalePrice", data=data)
        fig.axis(ymin=0, ymax=800000);
        plt.xticks(rotation=90);
        plt.show()

    def feature_transformation(self):
        train=self.train
        test = self.test
        all_data =self.all_data
        ntrain =self.ntrain
        ntest =self.ntest
        all_data['MSSubClass'].describe()
        print(all_data['MSSubClass'].describe())
        # MSSubClass =The building class
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

        # Changing OverallCond into a categorical variable
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)

        # Year and month sold are transformed into categorical features.
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)
        all_data['KitchenQual'].unique()
        print(all_data['KitchenQual'].unique())
        from sklearn.preprocessing import LabelEncoder
        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
                'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
                'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
                'YrSold', 'MoSold')
        # Process columns and apply LabelEncoder to categorical features
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(all_data[c].values))
            all_data[c] = lbl.transform(list(all_data[c].values))

        # Check shape
        print('Shape all_data: {}'.format(all_data.shape))
        # Adding Total Square Feet feature
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
        # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
        train["SalePrice"] = np.log1p(train["SalePrice"])

        # Check the new distribution
        sns.distplot(train['SalePrice'], fit=norm);
        #all_data = all_data[self.cols[2:]]
        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(train['SalePrice'])
        print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('SalePrice distribution')

        fig = plt.figure()
        res = stats.probplot(train['SalePrice'], plot=plt)
        plt.show()

        y_train = train.SalePrice.values

        print("Skewness: %f" % train['SalePrice'].skew())
        print("Kurtosis: %f" % train['SalePrice'].kurt())

        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skewed Features': skewed_feats})
        skewness.head()

        skewness = skewness[abs(skewness) > 0.75]
        print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        from scipy.special import boxcox1p
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            all_data[feat] = boxcox1p(all_data[feat], lam)
            all_data[feat] += 1
        all_data = pd.get_dummies(all_data)
        print(all_data.shape)
        ntrain = train.shape[0]
        ntest = test.shape[0]

        y_train = train.SalePrice.values
        train = pd.DataFrame(all_data[:ntrain])
        test = pd.DataFrame(all_data[ntrain:])
        self.transformedtrain=train
        self.transformedtest =test
        self.y_train = y_train


    def results(self):
        y_valid= self.y_valid
        preds= self.preds
        model_name = self.model_name
        rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
        print(model_name,":rms",rms)
        score= r2_score(y_valid, preds)
        print(model_name, ":score", score)
        mae= mean_absolute_error(y_valid, preds)
        print(model_name, ":mae", mae)
        mse= mean_squared_error(y_valid, preds)
        print(model_name, ":mse", mse)
        pearson_coef, p_value = stats.pearsonr(y_valid, preds)
        print(model_name, ":pearson_coef", pearson_coef)
        print(model_name, ":p_value ", p_value )
        # Plot Histogram

        sns.distplot(preds, fit=norm);
        # Get the fitted parameters used by the function
        (mu1, sigma1) = norm.fit(preds)
        print(model_name,'\n Predicted mu = {:.2f} and sigma = {:.2f}\n'.format(mu1, sigma1))
        plt.legend(['Predicted Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],
                   loc='best')
        plt.ylabel('Frequency')
        plt.title('{m} Predicted SalePrice distribution'.format(m=model_name))
        # Plot Histogram
        sns.distplot(y_valid, fit=norm);
        sns.distplot(preds, fit=norm);
        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(y_valid)
        (mu1, sigma1) = norm.fit(preds)
        print(model_name,'\n Valid mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
        print(model_name,'\n Predicted mu = {:.2f} and sigma = {:.2f}\n'.format(mu1, sigma1))
        plt.legend(['Valid Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
        plt.legend(['Predicted Normal dist. ($\mu1=$ {:.2f} and $\sigma1=$ {:.2f} )'.format(mu1, sigma1)], loc='best')
        plt.ylabel('Frequency')
        plt.title('{m} Valid vs Predicted SalePrice distribution'.format(m=model_name))

        fig = plt.figure()
        res = stats.probplot(y_valid, plot=plt)
        res = stats.probplot(preds, plot=plt)
        plt.title('{m} Valid vs Predicted Probability Plot'.format(m=model_name))
        plt.show()

    def linear_regression(self):
        l = len(self.transformedtrain.values)-448
        print(l)
        x_train = self.transformedtrain.values[:l,:]
        x_test =self.transformedtest.values
        x_valid =self.transformedtrain.values[l:]
        y_train = self.y_train[:l]
        y_valid = self.y_train[l:]
        print("size",x_train.shape)
        # implement linear regression

        model = LinearRegression()
        model.fit(x_train, y_train)
        # make predictions and find the rmse
        preds = model.predict(x_valid)
        a=model.score(x_valid,y_valid)
        print("Linear Regression Score",a)
        self.y_valid = y_valid
        self.preds = preds
        self.model_name = "Linear Regression"

    def k_nearest_neighbor(self):
        l = len(self.transformedtrain.values) - 448
        print(l)
        x_train = self.transformedtrain.values[:l, :]
        x_test = self.transformedtest.values
        x_valid = self.transformedtrain.values[l:]
        y_train = self.y_train[:l]
        y_valid = self.y_train[l:]
        # using gridsearch to find the best parameter
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
        knn = neighbors.KNeighborsRegressor()
        model = GridSearchCV(knn, params, cv=5)

        # fit the model and make predictions
        #a=model.score(x_valid,y_valid)
        #print(" K nearest Neighbor Score",a)
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        self.y_valid = y_valid
        self.preds = preds
        self.model_name = "K nearest Neighbor"





if __name__ == "__main__":
    House_Price_Prediction_System = House_Price_Prediction_System()
    House_Price_Prediction_System.data_acquisition()
    House_Price_Prediction_System.saleprice_variable_statistical_analysis()
    House_Price_Prediction_System.multivariable_feature_analysis()
    House_Price_Prediction_System.treat_missing_values_outlier()
    House_Price_Prediction_System.feature_transformation()
    House_Price_Prediction_System.linear_regression()
    House_Price_Prediction_System.results()
    House_Price_Prediction_System.k_nearest_neighbor()
    House_Price_Prediction_System.results()
    House_Price_Prediction_System


