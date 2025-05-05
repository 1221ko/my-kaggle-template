

- kaggle CL sample code
  - submit 
    - kaggle competitions submit -c titanic -f -m "comments"
    - -c: competition name
    - -f: file name
    - -m: message 
  - confirm score
    - kaggle competitions submissions -c titanic



- 学び
  - 01: titanic 
    - sum(hoge) / len(hoge)で割合を求める
    - pd.getdummies(df)でカテゴリカルなカラムもまとめて学習データに
  - 02: store-sales時系列モデル
    - .resample("D"): 漏れがある日付データを連続に埋める
    - .interpolate(): 内挿する
    - 