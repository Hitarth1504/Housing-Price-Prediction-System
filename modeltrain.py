import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("finaldatset.csv")

x = df[['ocean_proximity_Encoded','longitude_scal','latitude_scal','housing_median_age_scal','total_rooms_scal','total_bedrooms_scal','population_scal','households_scal','median_house_value_scal']]
y = df['median_income_scal']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)

print(x_train)
print(y_test)

Model = LinearRegression()

Model.fit(x_train,y_train)
score = Model.predict(x_test)
r2 = r2_score(y_test,score)
print(r2)