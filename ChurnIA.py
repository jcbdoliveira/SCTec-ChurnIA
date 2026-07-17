import sys
import os

import src.config as Config
import src.install as Install


def main(argumento):
    vTreino= 'v1'

    if argumento == "--treinar" or argumento == "--prever":
        #==========================================================================
        #Antes de executar o script cria toda a estrutura de pastas e 
        #arquivos necessários para o projeto
        #==========================================================================        
        print("==================================================================")
        print(f"Criando pastas")
        print("==================================================================")
        Config.criar_pastas() 
        print("==================================================================")
        print(f"Instalando bibliotecas")
        print("==================================================================")
        Install.instalar_bibliotecas()
        #==========================================================================

    import src.pipeline as Pipeline

    if argumento == "--treinar":
        print("==================================================================")
        print("Treinamento de ML para previsão de CHURN")
        print("==================================================================")
        Pipeline.treinar()

    elif argumento == "--prever":
        print("==================================================================")
        print("Previsão de CHURN com modelo treinado")
        print("==================================================================")
        Pipeline.prever()

    else:
        print("Argumento inválido(--treinar ou --prever)")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Argumento inválido(--treinar ou --prever)")
        sys.exit(1)

    argumento = sys.argv[1]
    main(argumento)