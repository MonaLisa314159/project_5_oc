#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:28:51 2024

@author: mona
"""

# Fichier test_app.py utilisant Pytest
import pytest
from app import create_app 
from flask.testing import FlaskClient
import torch
import transformers
from transformers import BertTokenizer 
import joblib 


@pytest.fixture()
def app():
    app = create_app()
    app.config.from_mapping({"TESTING": True})
    yield app

@pytest.fixture()
def client(app):
    with app.test_client() as client:
        yield client

@pytest.fixture()
def model():
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
            self.l2 = torch.nn.Dropout(0.3)
            self.l3 = torch.nn.Linear(768, 100)
    
        def forward(self, ids, mask, token_type_ids):
            _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output

    model_instance = BERTClass()  # Créer une instance du modèle
    checkpoint = torch.load('bert_model.pth', map_location=torch.device('cpu'))
    model_instance.load_state_dict(checkpoint['model_state_dict'])  # Chargez les poids du modèle depuis le fichier
    model_instance = model_instance.to(torch.device('cpu'))
    model_instance.eval()  

    return model_instance

@pytest.fixture()
def tokenizer():
    bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer', map_location=torch.device('cpu'))
    return bert_tokenizer

@pytest.fixture
def mlb():
    mlb = joblib.load('mlb_model.joblib')
    return mlb

# 1.  Test de la création de l'application Flask :
def test_create_app(app):
   assert app is not None
    
#2.  Test verification que la route attendue existe
def test_routes_exist(client):
    response = client.get('/')
    assert response.status_code == 200


# 3. Test du chargement du fichier MultilabelBinazer
def test_load_multilabel_binarizer(mlb):
    assert mlb is not None
    assert hasattr(mlb, 'classes_') and len(mlb.classes_) == 100


# 4. Test de la prédiction de tags :
def test_prediction(client):
    question = "how use deep-learning natural language processing and multi-label classification"
    response = client.post('/', data={'question': question})
    assert response.status_code == 200

# 5. Test de l'intégration du modèle et du tokenizer et de la prédiction des tags pour une question
def test_model_integration(model, tokenizer):
    question = "how use deep-learning natural language processing and multi-label classification"
    inputs = tokenizer(question, return_tensors='pt')
    output = model(ids=inputs['input_ids'], mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
    probabilities = torch.nn.functional.sigmoid(output)
    assert output is not None
    assert probabilities is not None
    
# 6. Test des seuils de prédiction :
def test_threshold(model, tokenizer, mlb):
    question = "how use deep-learning natural language processing and multi-label classification"
    inputs = tokenizer(question, return_tensors='pt')
    output = model(ids=inputs['input_ids'], mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
    probabilities = torch.nn.functional.sigmoid(output)
    threshold = 0.6
    predictions_array = (probabilities.detach().numpy() >= threshold).astype(int)
    target_names = mlb.classes_  
    predictions = [
                {'class': class_name, 'probability': float(probabilities[0, i])}
                for i, class_name in enumerate(target_names) if predictions_array[0, i] == 1
                  ]
    assert predictions is not None
    


