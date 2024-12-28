import pandas as pd

df = pd.read_excel(r"C:\Users\16199\Downloads\wjp_lr_HS_for.01.txt.xlsx")

print(list(df.iloc[16,:]))