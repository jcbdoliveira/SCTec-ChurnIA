import os

def criar_pastas():
    # Lista de pastas a serem criadas
    pastas = [
        "src",
        "data",
        "data/raw",
        "data/processed",
        "data/final",
        "models",
        "models/v1",
        "outputs",
        "outputs/figures"        
    ]

    # Criação das pastas
    for pasta in pastas:
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f"Pasta '{pasta}' criada com sucesso.")
        else:
            print(f"Pasta '{pasta}' já existe.")