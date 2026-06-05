import pickle
from model import load_data, features

df = load_data('./cache/historical/RELIANCE.csv')

model = pickle.load(open('./models/model.pkl', 'rb'))

last_row = df[features].iloc[[-1]]
predicted_price = model.predict(last_row)

print("predicted next day close price: ₹", round(predicted_price[0], 2))