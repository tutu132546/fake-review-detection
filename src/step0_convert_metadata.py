import pandas as pd
from config import RAW_TXT_PATH, RAW_CSV_PATH

def main():
    print("正在将 metadata.txt 转换为 YelpChi_reviews.csv...")
    df = pd.read_csv(RAW_TXT_PATH, sep=r'\s+', header=None, 
                     names=['user_id', 'product_id', 'rating', 'label', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(RAW_CSV_PATH, index=False)
    # 去掉这里的 Emoji 符号
    print(f"转换完成，文件已保存至 {RAW_CSV_PATH}，共 {len(df)} 条。")

if __name__ == "__main__":
    main()