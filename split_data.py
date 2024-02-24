from sklearn.model_selection import train_test_split
import pandas as pd

def auto_splitting(data, company, test_p = 0.20):
    df_train, df_test = train_test_split(data, 
                                         test_size=test_p, 
                                         random_state=2024)  
    
    df_train.to_parquet(f'data/{company.lower()}/{company.lower()}_train.parquet')
    df_test.to_parquet(f'data/{company.lower()}/{company.lower()}_test.parquet')
    
lyft_full = pd.read_parquet("data/lyft/lyft_full_data.parquet")
uber_full = pd.read_parquet("data/uber/uber_full_data.parquet")

print(auto_splitting(lyft_full, company="lyft"))
print(auto_splitting(uber_full, company="uber"))