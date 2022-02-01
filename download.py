from src.data import download_reports, extract_sa_fcas_data

download_reports("https://nemweb.com.au/Reports/Current/Public_Prices/")

print("Extracting FCAS data... ", end="")

df = extract_sa_fcas_data()
df.reset_index(drop=True, inplace=True)
df.to_csv("data/sa_fcas_data.csv", index=False)

print("done.")
