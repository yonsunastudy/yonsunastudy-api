from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from pathlib import Path
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
""" from transformers import BertTokenizer, BertModel """
from transformers import AutoModel, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import numpy as np

# FastAPIインスタンス
app = FastAPI()

# 日本語BERTの準備
print("現在、正誤判定用のモデル「model_izumi-lab--bert-small-japanese」を読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained("./model_izumi-lab--bert-small-japanese")
model = AutoModel.from_pretrained("./model_izumi-lab--bert-small-japanese")

""" model_name = "model_izumi-lab--bert-small-japanese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name) """

print("モデルの読み込みは正常に完了しました。")

def get_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**tokens)

    embeddings = output.last_hidden_state.mean(dim=1)  # 平均プーリング

    return embeddings.squeeze().numpy()


# 類似度を計算
def calculate_similarity(text1, text2):
    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)
    similarity = 1 - cosine(vec1, vec2)  # コサイン類似度
    return similarity


class JudgmentRequest(BaseModel):
    correctAnswer: str
    userAnswer: str
    requiredKeywords: list[str]


# APIエンドポイント
@app.post("/api/judgment/")
async def judgment(item: JudgmentRequest):
    dump = item.model_dump()
    user_answer = dump["userAnswer"]
    correct_answer = dump["correctAnswer"]
    required_keywords = dump["requiredKeywords"]
    joined_keywords = ", ".join(required_keywords)

    # 類似度を計算
    similarity = calculate_similarity(user_answer, correct_answer)
    all_keywords_present = all(keyword in user_answer for keyword in required_keywords)
    some_keywords_present = any(keyword in user_answer for keyword in required_keywords)

    # 判定
    if similarity > 0.9 and all_keywords_present:
        judgmentResults = {
            "judgment": "◯",
            "status": 0,
        }

        grading_criteria = f"良く記述ができており、必要なキーワード（{joined_keywords}）が全て含まれています。これからも頑張ってください。"
    elif similarity > 0.6 and some_keywords_present:
        judgmentResults = {
            "judgment": "△",
            "status": 1,
        }

        grading_criteria = (
            "少し惜しい点があるようです。模範解答と比較してみてください。"
        )

        if similarity > 0.9 and not all_keywords_present:
            similarity = 0.6

            grading_criteria = f"少し惜しい点があります。必要なキーワード（{joined_keywords}）が全て含まれていないようです。模範解答と比較してみてください。"
    else:
        judgmentResults = {
            "judgment": "✕",
            "status": 2,
        }

        grading_criteria = f"あなたの解答は模範解答と違うようです。何が違うのかや問題をもう一度見直してみてください。また、この記述では次のキーワード「{joined_keywords}」を必ず含める必要があります。"

        if not some_keywords_present:
            similarity = 0

    return jsonable_encoder({
        "answer": {
            "currentAnswer": correct_answer,
            "userAnswer": user_answer,
        },
        "requiredKeywords": required_keywords,
        "gradingCriteria": grading_criteria,
        "similarity": round(float(similarity), 2),
        "results": judgmentResults,
    })

@app.get("/favicon.ico")
async def favicon():
    image_path = Path("./favicon.ico")
    return FileResponse(image_path)


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(_: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"error": "お探しのページは見つかりませんでした。"},
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"エラーが発生しました: {exc.detail}"},
    )
