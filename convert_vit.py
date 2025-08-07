import pandas as pd
import os

def convert_csv(input_csv, image_dir, output_csv):
    print(f"📄 処理中: {input_csv}")
    
    # CSV読み込み
    df = pd.read_csv(input_csv)
    
    # クラス列を特定（filename と cloud を除く）
    class_cols = [col for col in df.columns if col not in ["filename", "cloud"]]
    
    # 数値型に変換（1や0が文字列の場合に対応）
    df[class_cols] = df[class_cols].apply(pd.to_numeric, errors="coerce")
    
    # ラベル列を作成（最大値が1の列名）
    df["label"] = df[class_cols].idxmax(axis=1)
    
    # 画像パスを追加
    df["image"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))
    
    # 中間チェック
    print("▶️ サンプル（先頭5件）:")
    print(df[["image", "label"]].head())
    
    # 出力
    df_out = df[["image", "label"]]
    df_out.to_csv(output_csv, index=False)
    print(f"✅ {output_csv} を作成しました\n")

# 実行例
convert_csv("train.csv", "train", "train_converted.csv")
convert_csv("valid.csv", "valid", "valid_converted.csv")
convert_csv("test.csv", "test", "test_converted.csv")
