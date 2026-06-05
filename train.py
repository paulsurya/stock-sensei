import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from model import load_data, features

df = load_data('./cache/historical/RELIANCE.csv')

x = df[features]
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

pickle.dump(model, open('./models/model.pkl', 'wb'))
print("model saved as model.pkl")