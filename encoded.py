from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cleandata.csv")

df_label = df.copy()
le=LabelEncoder()

df_label['ocean_proximity_Encoded'] = le.fit_transform(df['ocean_proximity'])
print(df_label)

scal = [
    'longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value'
]

sca = StandardScaler()
scal_data = sca.fit_transform(df[scal])
scall_data = pd.DataFrame(scal_data,columns=scal)

df_final = pd.concat([df_label,scall_data],axis=1)
df_final.to_csv("finaldatset.csv",index=False)

