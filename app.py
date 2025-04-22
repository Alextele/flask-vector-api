from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
from llama_cpp import Llama
import config

app = Flask(__name__)

# Для /vector
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

# Для /generate
llm = Llama(
    model_path="models/saiga_yandexgpt_8b.Q4_K_M.gguf",  # Обновлённое имя модели
    n_ctx=8192,
    n_parts=1,
    verbose=True,
)

DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


def get_vector(question):
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector.tolist()


@app.route('/vector', methods=['POST'])
def vector_endpoint():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required!'}), 400
    vector = get_vector(question)
    return jsonify({'vector': vector})


@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.get_json()
    question = data.get("question", "")
    context = data.get("context", [])
    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # Параметры генерации с возможностью переопределения
    temperature = data.get("temperature", 0.6)
    top_k = data.get("top_k", 30)
    top_p = data.get("top_p", 0.9)
    repeat_penalty = data.get("repeat_penalty", 1.1)
    max_tokens = data.get("max_tokens", 512)
    stop = data.get("stop", ["\nUser:", "\nSystem:", "\nАссистент:", "\nAssistant:", "###", "```", "Вопрос:", "?" ])

    if not question:
        return jsonify({'error': 'Question is required!'}), 400

    if context:
        first = context[0]
        rest = context[1:] if len(context) > 1 else []

        context_text = (
            "Отвечай строго по приведённым утверждениям. Не искажай смысл.\n\n"
            f"1. Начало ответа должно передавать смысл следующего утверждения: \"{first}\"\n"
        )

        if rest:
            context_text += (
                    "2. Далее последовательно используй следующие утверждения, если они соответствуют вопросу:\n"
                    + "\n".join(f"- {item}" for item in rest)
            )

        context_text += (
            "\n\n Жёсткие правила:\n"
            "- Не используй внешнюю информацию.\n"
            "- Не добавляй других слов и знаний.\n"
            "- Каждое утверждение несёт самостоятельный смысл.\n"
            "- Не мешай уверждения между собой в общие фразы. Одно утверждение - одно предложение в ответе.\n"
            "- Не объединяй утверждения для формирования ответа.\n"
            "- В ответе используй последовательно каждое утверждение, если оно несет в себе уникальный смысл.\n"
            "- Не придумывай, не дополняй, не обобщай.\n"
            "- Не делай предположений.\n"
            "- Не используй форматирование (жирный, курсив, Markdown и т.п.) в ответах.\n"
            "- Отвечай кратко, не делая выводов, и не добавляя от себя ничего."
        )

        user_message = f"{context_text}\nВопрос: {question}"
    else:
        user_message = question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    result = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        max_tokens=max_tokens,
        stop=stop,
        stream=False
    )

    reply = result["choices"][0]["message"]["content"].strip()
    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=True)
