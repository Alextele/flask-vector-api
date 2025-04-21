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
    stop = data.get("stop", None)

    if not question:
        return jsonify({'error': 'Question is required!'}), 400

    if context:
        first = context[0]
        rest = context[1:] if len(context) > 1 else []
        context_text = f"Обязательно начни ответ с этой информации: {first}"
        if rest:
            context_text += "\nПродолжи, используя эту информацию:\n- " + "\n- ".join(rest) + "\nВажно: Не используй более никакой информации при формировании ответа!"
        user_message = f"{context_text}\n\nВопрос: {question}"
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
