from src.data import download_reports, extract_sa_fcas_prices

download_reports("https://www.nemweb.com.au/REPORTS/CURRENT/Daily_Reports/")
download_reports("https://nemweb.com.au/Reports/Current/Settlements/")

df = extract_sa_fcas_prices()
df.reset_index(drop=True, inplace=True)
df.to_csv("data/sa_fcas_prices.csv")
