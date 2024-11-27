from transformers import AutoModel, AutoTokenizer

# モデル名（事前ダウンロード用）
model_name = "cl-tohoku/bert-base-japanese"

# モデルとトークナイザーをロード
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# モデルをローカルディレクトリに保存
tokenizer.save_pretrained("./model")
model.save_pretrained("./model")

print("モデルの事前ダウンロードが完了しました")
