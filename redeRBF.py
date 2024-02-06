from matplotlib.patches import Circle
from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as pl
import seaborn as sns

from sklearn.linear_model import LogisticRegression

#   Importa tabela e Separar em conjunto de Treinamento

camadas = 3
variaveis_iniciais = 2
nucleos_de_saida = 1
nucleos_intermediarias = 2
maximo_de_epoca = 1e6
taxa_aprendizagem = 0.1
precisao_requerida = 1e-7 # 1*10^-7
def importTable() -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    dataEx = pd.read_excel('data_RBF.xls', sheet_name='Plan1', index_col=0)
   
    df1 = dataEx.sample(frac=0.9, random_state=42)  # randomly select 90% of the rows
    df2 = dataEx.drop(index=df1.index)  # remove the selected rows from the original dataframe
    
    return df1, df2
 
# Implementar função de Base Radial 
def rbf_kernel(x, center, variance, gamma=1.0):
    
    x = np.array(x)
    center = np.array(center)
    distance = np.linalg.norm(x - center)
    kernel_value = np.exp(-gamma * distance ** (2 * variance ** 2))
    return kernel_value

def calc_variancia(tabP: pd.DataFrame, centers: np.ndarray) -> np.ndarray:
    
    variances = np.zeros(len(centers))
    cluster_counts = tabP["group_cluster"].value_counts()

    for j in range(len(centers)):
        sum_squared_distance = 0
        instances_in_cluster = tabP[tabP["group_cluster"] == j]

        for _, instance in instances_in_cluster.iterrows():
            squared_distance = np.sum((instance[["x1", "x2"]] - centers[j]) ** 2)
            sum_squared_distance += squared_distance

        variances[j] = sum_squared_distance / cluster_counts[j]

    return variances

def adicionando_variancia_ao_plot(centers  , variancia_calculada) -> Tuple[Circle, Circle]:
        # Plot cluster centers as blue and green crosses
    colors = ['b', 'g']
    for i, center in enumerate(centers):
        pl.plot(center[0], center[1], colors[i]+'+')

    # Create circles for each cluster using the cluster centers and variances
    circles = []
    for i, center in enumerate(centers):
        circle = Circle((center[0], center[1]), variancia_calculada[i], fill=False)
        circles.append(circle)

    # Add circles to the plot
    for circle in circles:
        pl.gca().add_patch(circle)

    # Return the created circles
    return tuple(circles)
    

def iniciar_matriz_peso_sinapticos(x, y):
    matriz = [[round(np.random.uniform(-1, 1), 7) for _ in range(y + 1)] for _ in range(x)]
    return matriz

def funcao_de_ativacao( valor):
    # if(select == "linear"):
    #     linear = 1 if valor >= 0 else -1
        return valor
        
        
def atualizando_npArray(Array, coluna):
    if Array.shape[1] < 5:
        
        Array = np.insert(Array, 4, coluna, axis=1)

    else:
        Array[:, 4] = coluna

    # print(Array) 
    return Array

def calculando_valor_saida(Array_copy, matriz_copy, centers_copy):
            
    coluna_Y_obtido = np.array([]) 
    for i in Array_copy:
        # X_array[i][0] # x1
        # X_array[i][1] # x2
        # X_array[i][2] # d
        # X_array[i][3] # r-cluster
        # print(f"i:  {i} ")
        
        j_b = -1*(matriz_copy[0][0]) 
        j_0 = rbf_kernel((i[0],i[1]),(centers_copy[0][0],centers_copy[0][1]),variancia_calculada[0])*matriz_copy[0][1]
        j_1 = rbf_kernel((i[0],i[1]),(centers_copy[1][0],centers_copy[1][1]),variancia_calculada[1])*matriz_copy[0][2]
        sum_j = j_b + j_0 + j_1
        Y_i = [j_b, j_0, j_1]
        
        result_g1 = funcao_de_ativacao(valor = sum_j )
        # print (f"result_g1 :  {result_g1}")
        
        # criando um lista para transforma em coluna na tabela 
        coluna_Y_obtido = np.append(coluna_Y_obtido, result_g1)
    
    # print (f"coluna_Y_obtido tam( { len(coluna_Y_obtido) } ) : {coluna_Y_obtido}")
    return coluna_Y_obtido, Y_i 
       
def calculando_erro_medio(Array_copy):
    sum_erro = 0
    for coluna in Array_copy:
        erro = 0
        erro =   ((coluna[2] - coluna[4])**2 ) / 2
        sum_erro += erro
        
        # divide o somatorio do erro pela quantidade de atributos
        # erro_medio = sum_erro / len(coluna_Y_obtido)
    erro_medio = sum_erro / len(coluna_Y_obtido) 
    return erro_medio
if __name__ == '__main__':
    print("Iniciando Variaveis")
    for i in [("camadas", camadas), ("variaveis_iniciais", variaveis_iniciais), ("nucleos_de_saida", nucleos_de_saida), ("nucleos_intermediarias", nucleos_intermediarias), ("taxa_aprendizagem", taxa_aprendizagem), ("precisao_requerida", precisao_requerida)]:
        print(f"{i[0]}: {i[1]}\n")

    print(f" Importando tabela: ")
    data = importTable()
    treinamento = data[0]
    teste = data[1]
    print(f"Formatando Dados: ")
    treinamento_com_index_recetado = pd.DataFrame( treinamento ).reset_index()
    testeComIndexRecetado = pd.DataFrame( teste ).reset_index()
        
    # Criando Coluna de dados para  K-Means 
    X_train = np.array(treinamento_com_index_recetado.drop('d',axis=1))
    
    # Etapa 01
    # Calculando os Centros
    kmeans = KMeans(n_clusters=nucleos_intermediarias, max_iter=300, init="random")
    kmeans.fit(X_train)

    centers = kmeans.cluster_centers_
    numeros_de_interações = kmeans.n_iter_
    k_labels_by_centers = kmeans.labels_

    # Iniciando Pesos Sinapticos    
    matrizW2 =  iniciar_matriz_peso_sinapticos(nucleos_de_saida,nucleos_intermediarias) # linhas, colunas

    # Adicionando a tabela -> COLUNA  com referencia aos Clusters
    treinamento_com_index_recetado["group_cluster"] = k_labels_by_centers
    # print (treinamento_com_index_recetado)
    sns.relplot(data=treinamento_com_index_recetado, x="x1", y="x2", hue="group_cluster")
    # modificando a tabela para adicionar as labels de dos clusters 
    variancia_calculada =  calc_variancia(tabP = treinamento_com_index_recetado, centers = centers)
    
    # adicionar center clusters Ao plot Com raios
    adicionando_variancia_ao_plot(centers,variancia_calculada)

    # print(treinamento_com_index_recetado)
    
    # Agora pego os valores e aplico para ver os resultados 
    X_array = np.array(treinamento_com_index_recetado)
    # print(type (X_array) )
    
    
    coluna_Y_obtido, Y_i = calculando_valor_saida(Array_copy=X_array, centers_copy= centers,matriz_copy=matrizW2)
    X_array = atualizando_npArray(X_array,coluna_Y_obtido)
    
    # Iniciar contador de Epocas
    epoca = 0    
    erro_medio_anterior = 0 
    erro_medio_atual = 0
    condition = False
    

    aux = 1    
    while True:
        # Calcular o Erro medio -> Atribuir ao erro_medio_anterior
        #    adiciona a coluna ao NpArray

        # X_array =  np.insert(X_array,4, coluna_Y_obtido, axis=1)
        # print(X_array)
        
        X_array = atualizando_npArray(Array=X_array, coluna=coluna_Y_obtido)
            # faz a subtração da entre as colunas

        erro_medio_anterior = calculando_erro_medio(Array_copy=X_array)
        # Ajustar W_2ji e "O" com os passos do PMC
        #       δ = desejado - obtido
             
        for coluna in X_array:
            saida_desejada = coluna[2]
            saida_obtida = coluna[4]
            sigma = (saida_desejada - saida_obtida)*1
            # print(f"W_ij antes: {matrizW2} ")
            index = 0
            for j in matrizW2[0]:
                Y_i_local =  Y_i[index]
                j += taxa_aprendizagem * sigma * Y_i_local
                matrizW2[0][index] = j
                index += 1
                # print(f"valor de j ->  {j} ")
        # print(f"W_ij depois: {matrizW2}")
        # Rodar novamente com a nova matriz W_2ij
        coluna_Y_obtido,Y_i_local = calculando_valor_saida(Array_copy=X_array, centers_copy= centers,matriz_copy=matrizW2)
        X_array = atualizando_npArray(X_array,coluna_Y_obtido)
        # atualizar erro_Medio
        erro_medio_atual = calculando_erro_medio(Array_copy=X_array) 
        # Atualizar epoca
        
        
        # Condição de parada
        condition = True if (abs( erro_medio_atual - erro_medio_anterior ) <= precisao_requerida) else False
        dif_erro =       abs( erro_medio_atual - erro_medio_anterior )
        
        print(f"Dif_erro : {dif_erro} --> {erro_medio_atual}  \n epoca : {epoca}") 
        if dif_erro != aux:
            print(f"dif_erro : {dif_erro} ")
        aux = dif_erro
        epoca += 1    
        if  condition   |  (epoca >= maximo_de_epoca) : # (erro_medio_atual <= 0.01)
            break;
    
    # Apuração dos resultados 
    
    
    print (f"Pesos: {matrizW2}")
    print(f"Tabela: {X_array} ")
    # Retorna os Valores obtidos
    
    
        
        