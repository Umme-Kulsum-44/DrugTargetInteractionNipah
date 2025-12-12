#importing required libraries

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from flask import flash
from flask_material import Material
import warnings
import string
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import os
# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the pre-trained logistic regression model
# Load the trained model
clf = joblib.load("drug_target_model.pkl")

# Load ProtBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
model = AutoModel.from_pretrained("Rostlab/prot_bert")

# Function to compute molecular descriptors
from rdkit.Chem import Descriptors


# Function to compute molecular descriptors
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
        ]
    else:
        return [None, None, None, None]

# Function to encode protein sequence
def encode_protein(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Function to compute molecular formula
def get_molecular_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return CalcMolFormula(mol)
    else:
        return "Invalid SMILES"

# Function to display molecular structure
def display_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        print("Invalid SMILES string for structure rendering.")
        

app = Flask(__name__)
Material(app)
app.secret_key="dont tell any one"

@app.route('/')
def home():
    return render_template('login.html')
    # User is not loggedin redirect to login page


@app.route('/main')
def main():
    return render_template('index.html')
    # User is not loggedin redirect to login page


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/',methods=["POST"])
def login():
    if request.method == 'POST':
        username = request.form['id']
        password = request.form['pass']
        if username=='admin' and password=='admin':
            return render_template("index.html")
        else:
            flash("wrong password")
            return render_template("login.html")

# Function to predict drug-target interaction
def predict_interaction(drug_smiles, target_sequence):
    # Compute descriptors
    descriptors = compute_descriptors(drug_smiles)
    if None in descriptors:
        return "Invalid SMILES format"

    # Encode protein sequence
    protein_embedding = encode_protein(target_sequence)

    # Combine features
    input_features = np.hstack((descriptors, protein_embedding))

    # Debugging: Print computed feature values
    print(f"Computed Descriptors: {descriptors}")
    print(f"Protein Embedding Shape: {protein_embedding.shape}")
    print(f"Combined Feature Vector Shape: {input_features.shape}")

    # Padding to match feature length
    if input_features.shape[0] < clf.n_features_in_:
        padding = np.zeros((clf.n_features_in_ - input_features.shape[0],))
        input_features = np.hstack((input_features, padding))
    elif input_features.shape[0] > clf.n_features_in_:
        return f"Feature mismatch: Expected {clf.n_features_in_}, but got {input_features.shape[0]}"

    # Debugging: Check input tensor before prediction
    print(f"Final Input Tensor Shape: {input_features.shape}")

    # Predict interaction
    input_tensor = np.array(input_features).reshape(1, -1)
    prediction = clf.predict(input_tensor)

    return "Active" if prediction[0] == "Active" else "Inactive"

@app.route('/main', methods=["POST"])
def analyze():
    if request.method == 'POST':
        formula = request.form['formula']
        sequence = request.form['sequence']

        try:
            # Compute descriptors and embeddings
            descriptors = compute_descriptors(formula)
            protein_embedding = encode_protein(sequence)
            molecular_formula = get_molecular_formula(formula)

            # Combine features into a single vector
            input_features = np.hstack((descriptors, protein_embedding)).reshape(1, -1)

            # Predict activity
            predicted_class = predict_interaction(formula, sequence)
            predicted_probabilities = clf.predict_proba(input_features)



            # Generate molecular structure image
            mol = Chem.MolFromSmiles(formula)
            if mol:
                structure_image_path = os.path.join('static', 'structure.png')
                Draw.MolToFile(mol, structure_image_path)
            else:
                structure_image_path = None

            return render_template(
                'contact.html',
                Formula=molecular_formula,
                Activity=predicted_class,
                Probabilities=predicted_probabilities,
                res=1,
                structure_image=structure_image_path
            )

        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/healthz")
def health():
    return {"status":"ok"}
