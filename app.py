from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import config

app = Flask(__name__)

# Инициализация BERT токенизатора и модели для векторизации
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Инициализация DialoGPT токенизатора и модели для генерации
dialogpt_model_name = "tinkoff-ai/ruDialoGPT-medium"
dialogpt_tokenizer = AutoTokenizer.from_pretrained(dialogpt_model_name)
dialogpt_model = AutoModelForCausalLM.from_pretrained(dialogpt_model_name)
dialogpt_model.eval()

def get_vector(question):
    # Токенизация вопроса
    inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

    # Получение выходных данных из модели
    with torch.no_grad():
        outputs = model(**inputs)

    # Получаем вектор вопроса (используем вектор [CLS])
    vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return vector.tolist()  # Преобразуем в список для JSON

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
    context_list = data.get("context", [])
    max_length = data.get("max_length", 300)
    top_p = data.get("top_p", 0.95)
    top_k = data.get("top_k", 50)

    prompt = "Контекст:\n"
    for idx, entry in enumerate(context_list):
        prompt += f"{idx+1}. {entry}\n"
    prompt += f"\nВопрос: {question}\nОтвет:"

    input_ids = dialogpt_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = dialogpt_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=dialogpt_tokenizer.eos_token_id
        )

    reply = dialogpt_tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT, debug=True)