from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel
# import config

app = Flask(__name__)

# Инициализация токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)