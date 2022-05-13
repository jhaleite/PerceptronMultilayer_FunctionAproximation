# -*- coding: utf-8 -*-

import numpy as np 
import random
import copy

# -----------------------------------------------------------------------------------------------------------------
# LER DADOS DE TREINAMENTO 

arq = open('/home/joao/Desktop/Redes Neurais Artificiais/EPCs/EPC04/Dados_entrada_epc4.txt','r')
texto = arq.read().split("\n")           # Lê o arquivo .txt e separa as string por cada \n
arq.close()                              # Fecha o arquivo

for i in range(len(texto)):
    texto[i] = texto[i].split("\t")      # Separa as strings por \t(tab) e os coloca dentro de um anova lista
    for j in range(len(texto[i])):
        texto[i][j] = float(texto[i][j]) # Converte de formato string para float
#print(texto)
entradas = len(texto[1])                 # guarda o a quantidade de entradas
amostras = len(texto)                    # guarda o numero de amostras para cada entrada
#print(entradas)
#print(amostras)
texto = np.array(texto)
texto = texto.T
Dados = np.zeros((entradas,amostras),dtype=float)

for i in range(0,entradas):
    for j in range(0, amostras):
            Dados[i][j] = texto[i][j]
#print(Dados)

# FIM LEITURA DE DADOS

# -----------------------------------------------------------------------------------------------------------------
# FASE DE TREINAMENTO
# -----------------------------------------------------------------------------------------------------------------

# OBTENDO CONJUNTO DE AMOSTRAS (X) E AS SAIDAS DESEJADAS (d):

X = []
X0 = []
d = []
for i in range(0,amostras):
    X0.append(np.array(-1))
X.append(X0)
for i in range(0,entradas):
    if i != (len(Dados)-1):
        X.append(Dados[i])
    else:
        d.append(Dados[i])

X = np.array(X)
d = np.array(d[0])
X = X.T
Namostras = len(X)
Nentradas = len(X[0])

#print('d = ',d)
#print(len(d))
#print('X = ',X)
#print(len(X))

# INICIANDO OS AS MATRIZES DE PESOS SINAPTICOS DA REDE COM NUMEROS ALEATORIOS

W2 = np.zeros((10,4),dtype=float)  # Pesos sinapticos entre a camada de entrada e a camada escondida
W3 = np.zeros((1,11),dtype=float)  # Pesos sinapticos entre a camada escondida e a camada de saida

for i in range(0,10):
    for j in range(0,4):
        W2[i][j] = random.random()


for i in range (0,1):
    for j in range(0,11):
        W3[i][j] = random.random()

'''print('W3 = ',W3)
print('=-'*60)
print(len(W3))
print('=-'*30)
print(len(W3[0]))
print('=-'*30)'''

# ESPECIFICANDO MINHA TAXA DE APRENDIZAGEM (eta) E MINHA PRECISÃO REQUERIDA (E)

eta = 0.1
E = 0.000001
EQM = 1

# INICIANDO O CONTADOS DE EPOCAS

epoca = 0

# ---------------------------------------------------------------------------------------------------------------------------------
# VARIAVEIS ASSOCIADAS A CAMADA ESCONDIDA

NeuronioEscondida = 10                              # Quantidade de Neuronios na camada escondida
Yj2 = np.zeros((NeuronioEscondida+1,1),dtype=float) # Saida dos neuronios da camada escondida
Y2 = []
Ij2 = np.ones((NeuronioEscondida,1),dtype=float)    # Entrada dos neuronios da camada escondida
I2 = []
for i in range(0,Namostras):                        # Diz que para cada conjunto de amostras eu terei um vetor da dimensão do Yj2    
        Y2.append(Yj2)
#print(Y2[0])
for i in range(0,len(Y2)):                          # Adiciona -1 refente ao valor da limiar do neuronio
    for j in range(0,len(Y2[0])):
        if j == 0:
            Y2[i][j] = -1
#print(Y2[5])

for i in range(0,Namostras):                        # Diz que para cada conjunto de amostras eu terei um vetor da dimensao do Ij2
    I2.append(Ij2)
#print(len(I2))

Deltaj2 = np.zeros((NeuronioEscondida,1),dtype=float) # Gradiente local referente aos neuronios da camada escondida
Delta2 = []
for i in range(0,Namostras):
    Delta2.append(Deltaj2)
#print(len(Delta2[0]))

# ------------------------------------------------------------------------------------------------------------------------------------
# VARIAVEIS ASSOSSIADAS A CAMADA DE SAIDA

NeuronioSaida = 1                               # Quantidade de Neuronios na camada de saida
Yj3 = np.zeros((NeuronioSaida,1),dtype=float)   # Sáida do neuronio da camada de saida
Y3 = []
for i in range(0,Namostras):                    # Diz que para cada conjunto de amostras eu terie um vetor na dimensao de Yj3
    Y3.append(Yj3)
#print(len(Y3))

Ij3 = np.ones((NeuronioSaida,1),dtype=float)    # Entrada do neuronio referente a camada de saida
I3 = []
for i in range(0,Namostras):                    # Diz que para cada conjunto de amostras eu terie um vetor na dimensao de Ij3 
    I3.append(Ij3)
#print(len(I3))

Deltaj3 = np.zeros((NeuronioSaida,1),dtype=float) # Gradiente referente ao neuronio da camada de saida
Delta3 = []
for i in range(0,Namostras):
    Delta3.append(Deltaj3)
#print((Delta3[0]))
# ------------------------------------------------------------------------------------------------------------------------------------
# INICIALIZANDO OS VETORES DE ERRO:

Ek = np.zeros((Namostras,1),dtype=float) # Inicializando o erro ek como um array para calculos
Ek1 = np.zeros((Namostras,1),dtype=float) # Inicializando o erro ek como um array para calculos
EmAnt = 0
EmAtu = 0
#print(W2)
#print('=-'*60)
#print(W3)

# -----------------------------------------------------------------
# ETAPA FORWARD:
# -----------------------------------------------------------------

# Para todas as amostras, obter:
for k in range(0,Namostras):
    somatorio = 0
    #print('##'*30)
    # Obter I2 e Y2 referentes a camada escondida

    for j in range(0,len(W2)):
        
        for i in range(0,len(W2[0])):
            somatorio = somatorio + (W2[j][i] *  X[k][i])
        I2[k][j] = copy.copy(somatorio)
        somatorio = 0
        Y2[k][j+1] = copy.copy(1/(1+np.exp(-I2[k][j])))   # j+1 é para o programa não interferir na posição 0 que contem o valor do meu limiar.
    somatorio = 0
    I2[k] = copy.copy(I2[k])
    Y2[k] = copy.copy(Y2[k])
    #print(Y2[k])
    #print('=-'*30)
    # Obter I3 e Y3 referente a camada de saida

    for j in range(0,len(W3)):
        for i in range(0,len(W3[0])):
            somatorio = somatorio + (W3[j][i] * Y2[k][i])
        I3[k][j] = copy.copy(somatorio)
        somatorio = 0
        Y3[k][j] = copy.copy(1/(1+np.exp(-I3[k][j]))) # não é necessario colocar j+1, pos como ja esta na camada de saida, não preciso usar mais o valor do meu limiar.
        #print(Y3[k][j])
    somatorio = 0
    I3[k] = copy.copy(I3[k])
    Y3[k] = copy.copy(Y3[k])
    #print(Y3[k])
    # FIM ETAPA FORWARD

'''print(Y2[0])
print('=-'*60)
print(Y3[0])
print('=-'*60)'''




while EQM > E:
    # Atribuindo valores para o vetor Erro medio
    somatorio  = 0
    for k in range(0,Namostras):
        Ek[k] = 0
        for j in range(0,NeuronioSaida): 
            somatorio  = somatorio + np.power((d.T[k]-Y3[k][j]),2)
        Ek[k] = (1/2)*somatorio
        somatorio = 0
    for k in range(0,Namostras):  #  namostras é o numero de amostras
        somatorio = somatorio + Ek[k]
    EmAnt = (1/Namostras)*somatorio
    somatorio = 0

    # Para todas as amostras, obter:
    for k in range(0,Namostras):
        somatorio = 0
        
        # -----------------------------------------------------------------
        # ETAPA BACKWARD:
        # -----------------------------------------------------------------
        # Calculando meu Gradiente local para j-ésimo neuronio de saida
         
        for j in range(0,NeuronioSaida):
            Delta3[k][j] = (d[k] - Y3[k][j]) * ((1/(1+np.exp(-I3[k][j])))*(1-(1/(1+np.exp(-I3[k][j])))))
        Delta3[k] = copy.copy(Delta3[k])
        #print(Delta3)
        #print('=-'*30)
        # Reajustando minha matriz de peso sinaptico entre a camada escondida e a de saida

        for j in range(0,NeuronioSaida):
            for i in range(0,NeuronioEscondida+1):
                W3[j][i] = W3[j][i] + (eta * Delta3[k][j] * Y2[k][i])

        # Calcular meu Gradiente local para da camada escondida

        for n in range(0,NeuronioSaida):           # Quantidade de Neuronios da camada de saida
            somatorio = 0
            for j in range(0,NeuronioEscondida):   # Quantidade de Neuronios da camada escondida
                somatorio = somatorio + (Delta3[k][n] * W3[n][j])
                #print(somatorio)
            for j in range(0,NeuronioEscondida):
                Delta2[k][j] = somatorio * (((1/(1+np.exp(-I2[k][j])))*(1-(1/(1+np.exp(-I2[k][j])))))) # CASO DER ERRO NOS RESULTADOS, PRESTAR ATENCAO AQUI!!!
                #print(Delta2[k][j])
        Delta2[k] = copy.copy(Delta2[k])
        somatorio = 0   

        # Reajustar minha matriz de peso sinaptico entre a camada de entrada e a escondida

        for j in range(0,NeuronioEscondida):
            for i in range(0,Nentradas):
                W2[j][i] = W2[j][i] + (eta * Delta2[k][j] * X[k][i])

        # FIM ETAPA BACKWARD
        # -----------------------------------------------------------------
    
    # OBTER Y3 AJUSTADO:

    for k in range(0,Namostras):
        #print('-='*30)
        somatorio = 0
        # -----------------------------------------------------------------
        # ETAPA FORWARD DENOVO:
        # -----------------------------------------------------------------
        # Obter I2 e Y2 referentes a camada escondida

        for j in range(0,len(W2)):
            for i in range(0,len(W2[0])):
                somatorio = somatorio + (W2[j][i] *  X[k][i])
            I2[k][j] = somatorio
            somatorio = 0
            Y2[k][j] = 1/(1+np.exp(-I2[k][j]))   # j+1 é para o programa não interferir na posição 0 que contem o valor do meu limiar.
        I2[k] = copy.copy(I2[k])
        Y2[k] = copy.copy(Y2[k])

        # Obter I3 e Y3 referente a camada de saida

        for j in range(0,len(W3)):
            for i in range(0,len(W3[0])):
                somatorio = somatorio + (W3[j][i] * Y2[k][i])
            I3[k][j] = somatorio
            somatorio = 0
            Y3[k][j] = 1/(1+np.exp(-I3[k][j])) # não é necessario colocar j+1, pos como ja esta na camada de saida, não preciso usar mais o valor do meu limiar.
        I3[k] = copy.copy(I3[k])
        Y3[k] = copy.copy(Y3[k])
        #print(Y3[k])
        # FIM ETAPA FORWARD DENOVO
        # -----------------------------------------------------------------

    # ATUALIZANDO ERRO:

    for k in range(0,Namostras):
        Ek1[k] = 0
        for j in range(0,NeuronioSaida): 
            somatorio  = somatorio + np.power((d[k]-Y3[k][j]),2)
        Ek1[k] = (1/2)*somatorio
        somatorio = 0
    
    for k in range(0,Namostras):  #  namostras é o numero de amostras
        somatorio = somatorio + Ek1[k]
    EmAtu = (1/Namostras)*somatorio
    somatorio = 0

    # INCREMENTANDO QUANTIDADE DE EPOCAS:

    epoca += 1 

    # ATUALIZANDO VALOR DO ERRO QUADRATICO MEDIO:

    EQM = np.max(np.abs(EmAnt-EmAtu))
    #print('##'*30)
    if epoca >=300:
        break
    
print('Erro quadratico medio = ', EQM)
print('=-'*60)
print('Numero de epocas = ',epoca)
print('=-'*60)
print('A saida Y: ')
for i in range(0,len(Y3)):
    print(Y3[i])
print('=-'*60)
print(W2)
print('=-'*60)
print(W3)
# -----------------------------------------------------------------------------------------------------------------
# FIM DA FASE DE TREINAMENTO
# -----------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
# INICIO DA FASE DE OPERAÇÃO : VALIDAÇÃO DA REDE DE ACORDO COM O CONJUNTO DE TESTES
# -----------------------------------------------------------------------------------------------------------------

# Ler amostras de testes para a fase de operação:

'''del(texto)
del(Dados)
del(X0)
del(X)
del(d)
del(Yj2)
del(Y2)
del(Ij2)
del(I2)
del(Yj3)
del(Y3)
del(Ij3)
del(I3)

arq = open('/home/joao/Desktop/Redes Neurais Artificiais/EPCs/EPC04/Dados_Classificacao_epc4.txt','r')
texto = arq.read().split("\n")           # Lê o arquivo .txt e separa as string por cada \n
arq.close()                              # Fecha o arquivo

for i in range(len(texto)):
    texto[i] = texto[i].split("  ")      # Separa as strings por \t(tab) e os coloca dentro de um anova lista
    for j in range(len(texto[i])):
        if i == (len(texto)-1):
            del(texto[i])
        else:
            texto[i][j] = float(texto[i][j]) # Converte de formato string para float
#print(len(texto))
entradas = len(texto[1])                 # guarda o a quantidade de entradas
amostras = len(texto)                    # guarda o numero de amostras para cada entrada
#print(entradas)
#print(amostras)
texto = np.array(texto)
texto = texto.T
Dados = np.zeros((entradas,amostras),dtype=float)

for i in range(0,entradas):
    for j in range(0, amostras):
            Dados[i][j] = texto[i][j]
#print(Dados)

# FIM LEITURA DE DADOS

# OBTENDO CONJUNTO DE AMOSTRAS (X) E AS SAIDAS DESEJADAS (d):

X = []
X0 = []
d = []
for i in range(0,amostras):
    X0.append(np.array(-1))
X.append(X0)
for i in range(0,entradas):
    if i != (len(Dados)-1):
        X.append(Dados[i])
    else:
        d.append(Dados[i])

X = np.array(X)
d = np.array(d[0])
X = X.T
Namostras = len(X)
Nentradas = len(X[0])
#print(X)
#print(d)

# VARIAVEIS ASSOCIADAS A CAMADA ESCONDIDA

NeuronioEscondida = 10                              # Quantidade de Neuronios na camada escondida
Yj2 = np.zeros((NeuronioEscondida+1,1),dtype=float) # Saida dos neuronios da camada escondida
Y2 = []
Ij2 = np.ones((NeuronioEscondida,1),dtype=float)    # Entrada dos neuronios da camada escondida
I2 = []
for i in range(0,Namostras):                        # Diz que para cada conjunto de amostras eu terei um vetor da dimensão do Yj2    
        Y2.append(Yj2)
#print(Y2[0])
for i in range(0,len(Y2)):                          # Adiciona -1 refente ao valor da limiar do neuronio
    for j in range(0,len(Y2[0])):
        if j == 0:
            Y2[i][j] = -1
#print(Y2[5])

for i in range(0,Namostras):                        # Diz que para cada conjunto de amostras eu terei um vetor da dimensao do Ij2
    I2.append(Ij2)
#print(len(I2[0]))

# VARIAVEIS ASSOSSIADAS A CAMADA DE SAIDA

NeuronioSaida = 1                               # Quantidade de Neuronios na camada de saida
Yj3 = np.zeros((NeuronioSaida,1),dtype=float)   # Sáida do neuronio da camada de saida
Y3 = []
for i in range(0,Namostras):                    # Diz que para cada conjunto de amostras eu terie um vetor na dimensao de Yj3
    Y3.append(Yj3)
#print(len(Y3))

Ij3 = np.ones((NeuronioSaida,1),dtype=float)    # Entrada do neuronio referente a camada de saida
I3 = []
for i in range(0,Namostras):                    # Diz que para cada conjunto de amostras eu terie um vetor na dimensao de Ij3 
    I3.append(Ij3)
#print(len(I3))

# -----------------------------------------------------------------
# ETAPA FORWARD: Referente a fase de operação
# -----------------------------------------------------------------

# Para todas as amostras, obter:
for k in range(0,Namostras):
    somatorio = 0
    
    # Obter I2 e Y2 referentes a camada escondida

    for j in range(0,len(W2)):
        for i in range(0,len(W2[0])):
            somatorio = somatorio + (W2[j][i] *  X[k][i])
        I2[k][j] = somatorio
        somatorio = 0
        Y2[k][j] = 1/(1+np.exp(-I2[k][j]))   
    I2[k] = copy.copy(I2[k])
    Y2[k] = copy.copy(Y2[k])
    somatorio = 0

    # Obter I3 e Y3 referente a camada de saida

    for j in range(0,len(W3)):
        for i in range(0,len(W3[0])):
            somatorio = somatorio + (W3[j][i] * Y2[k][i])
        I3[k][j] = somatorio
        somatorio = 0
        Y3[k][j] = 1/(1+np.exp(-I3[k][j])) 
    I3[k] = copy.copy(I3[k])
    Y3[k] = copy.copy(Y3[k])
# FIM ETAPA FORWARD

print('=-'*60)
print('A saida Y: ')
for i in range(0,len(Y3)):
    print(Y3[i])

# Calculo do erro absoluto:
Eabs = []
for k in range(0,Namostras):
    Eabs.append(d[k]-Y3[k])

# Calculo do erro Relativo:
Erel = []
for k in range(0,Namostras):
    Erel.append(Eabs[k]/d[k])

# Calculo do erro relativo medio:
ERM = 0
somatorio = 0
for k in range(0,Namostras):
    somatorio = somatorio + Erel[k]
ERM = (somatorio/len(Y3))*100

print('=-'*60)
print('Erro relativo medio = ',ERM)

# Calculo da Variancia da saida Y3

somatorio = 0
media = 0
for i in range(0,len(Y3)):
    somatorio = somatorio + Y3[k]
media = somatorio/len(Y3)
somatorio = 0
for i in range(0,len(Y3)):
    somatorio = somatorio + ((Y3[k]-media)**2)
Variancia = 0
Variancia = somatorio/(len(Y3)-1)
print('=-'*60)
print('Varianvia = ',Variancia)'''
    





    
  



    











        


        
        
            
