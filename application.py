#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:57:02 2024

@author: mona
"""

# Importer les bibliothèques nécessaires
from flask import Flask, request, jsonify, render_template, session
#from dotenv import load_dotenv
#import os
import torch
import transformers
from transformers import BertTokenizer 
import joblib 
#import logging

load_dotenv()

# Charger le modèle et le tokenizer
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

model = BERTClass()  # Créer une instance de votre modèle
checkpoint = torch.load('bert_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])  # Charger les poids du modèle depuis le fichier
model = model.to(torch.device('cpu'))
model.eval()  

tokenizer = BertTokenizer.from_pretrained('bert_tokenizer', map_location=torch.device('cpu'))  # Charger le tokenizer depuis le fichier

# Charger mlb à partir du fichier
mlb = joblib.load('mlb_model.joblib')

# Configurer la journalisation
#logging.basicConfig(filename='app.log', level=logging.DEBUG)


def create_app():
    my_app = Flask(__name__, template_folder='templates')
    my_app.secret_key = "314159265**" 	
    #my_app.secret_key = os.getenv('SECRET_KEY')  # Accéder à la clé secrète à partir des variables d'environnement
    
    @my_app.route('/', methods=['GET', 'POST'])
    def home():
        predictions = None
        if request.method == 'GET':
            question = session.get('question', '')
            return render_template('index.html', question=question)
    
        elif request.method == 'POST':
            # Si le formulaire est soumis, récupérer la question
            question = request.form['question']
            
            # Ajout des journaux pour suivre l'exécution
            #logging.info(f"Received question: {question}")
            
            # Stocker la question dans la session Flask
            session['question'] = question
            
            # Tokenizer la question
            inputs = tokenizer(question, return_tensors='pt')
            
            # Utiliser le modèle pour prédire les probabilités des classes
            output = model(ids=inputs['input_ids'], mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
            probabilities = torch.nn.functional.sigmoid(output)
            
    		# Appliquer le seuil
            threshold = 0.59
            predictions_array = (probabilities.detach().numpy() >= threshold).astype(int)
    
            # Obtenir les noms des classes
            target_names = mlb.classes_  # s'assurer que mlb est défini correctement
    
            # Filtrer les prédictions pour ne retourner que celles qui dépassent le seuil
            predictions = [
                {'class': class_name, 'probability': float(probabilities[0, i])}
                for i, class_name in enumerate(target_names) if predictions_array[0, i] == 1
            ]
    
        return render_template('index.html', predictions=predictions, question=question)
      
    return my_app


my_app = create_app()

# Exécuter l'application Flask
if __name__ == '__main__':
	my_app.run(debug=True)
