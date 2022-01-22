from src.data import download_daily_reports, extract_sa_fcas_prices

download_daily_reports()

df = extract_sa_fcas_prices()
df.to_csv("data/sa_fcas_prices.csv")
