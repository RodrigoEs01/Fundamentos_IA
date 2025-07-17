#===========================
#   Introducción al uso de softmax y cross entropy loss en pytorch
#===========================
#   Rodrigo Espinosa Lopez
#   Fundamentos de IA
#   ESFM IPN 2025
#===========================

#   Módulos necesarios
#===========================
import torch
import torch.nn as nn
import numpy as np

#===========================
#   Modelo de Boltzmann
#   En termodinámica es la probabilidad de encontrar un sistema en algún
#   estado dada su energía y temperatura
#===========================
#
#           -> 2.0                      -> 0.65
#    Linear -> 1.0      -> Softmax      -> 0.25     -> CrossEntropy(y, y_hat)
#           -> 0.65                     -> 0.1
#
#    puntajes (logits)           probabilidades suma = 1.0
#===========================


#   Softmax aplica el modelo de distribución exponencial para cada elemento
#   normalizada con la suma de todas las exponenciales
#===========================
def softmax(x):
    return np.exp(x) / np.sum (np.exp(x), axis=0)

#   Vector en R3
#===========================
x = np.array([2.0, 1.0, 0.1])

#   Softmax de elementos del vector
#===========================
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor ([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)       #   Tomar softmax de los elementos en el eje 0
print('softmax torch:', outputs)

#   Cross-entropy loss, o log loss, mide el rendimiento de un modelo de clasificación
#   cuya salida es un valor de probabilidad entre 0 y 1
#   Se incrementa conforme la probabilidad diverge del nivel verdadero
#===========================
def cross_entropy (actual, predicted):
    EPS = 1e-15
    #   Limitar los valores a un mínimo EPS y máximo 1-EPS
    #=========================== 
    predicted = np.clip(predicted, EPS, 1 - EPS)
    
    #   Cálculo del rendimiento
    #===========================
    loss = -np.sum(actual * np.log(predicted))
    return loss         # / float(predicted.shape[0])

#===========================
#   Debe ser alguna de las opciones
#   if class 0: [100]
#   if class 1: [010]
#   if class 2: [001]
#===========================
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array ([0.1, 0.3, 0.6])
l1 = cross_entropy (Y, Y_pred_good)
l2 = cross_entropy (Y, Y_pred_bad)
print(f'Loss1 numpy: {11:.4f}')
print(f'Loss2 numpy: {12:.4f}')

#===========================
#   CrossEntropyLoss en PyTorch (aplica Softmax)
#   nn.LogSoftmax + nn.NLLLoss
#   NLLLoss = "negative log likelihood loss"
#===========================
loss = nn.CrossEntropyLoss()
#   loss(input, target)

#===========================
#   Objetivo es de tamaño nSamples = 1
#   Cada elemento tiene etiqueta de clase: 0, 1, 0 2
#   Y (=objetivo) contiene etiquetas de clase class no opciones binarias
#===========================
Y = torch.tensor([0])

#   input es de tamaño nSamples x nClasses = 1 x 3
#   y_pred (input) deben estar sin normalizar (logits) para cada clase, no con softmax
#===========================
Y_pred_good = torch.tensor ([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])  

#   Usar loss = nn.CrossEntropyLoss()
#===========================
l1 = loss (Y_pred_good, Y)
l2 = loss (Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

#   Predicciones (regresa el máximo en la la dimensión)
#===========================
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

#   Permite calcular el rendimiento para múltiples conjuntos de datos
#   Vector objetivo es de tamaño nBatch = 3
#   cada elemento tiene etiqueta de clase: 0, 1, or 2
#===========================
Y = torch.tensor ([2, 0, 1])

#   Matriz input es de tamaño nBatch x nClasses = 3 x 3
#   Y_pred son logits (no softmax)
#===========================
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9],   #   Predice clase 2 
    [1.2, 0.1, 0.3],    #   Predice clase 0
    [0.3, 2.2, 0.2]])   #   Predice clase 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
    [0.1, 0.3, 1.5], 
    [1.2, 0.2, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Batch Loss1: {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

#   Predicciones
#===========================
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'clase verdadera: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')

#   Clasificación binaria (red neuronal)
#===========================
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        #   Sigmoide al final
        #===========================
        y_pred = torch.sigmoid(out)
        return y_pred

#correr problema de clasificación binaria
#===========================
model = NeuralNet1(input_size = 28*28, hidden_size = 5)
criterion = nn.BCELoss()

# Múltiples clases
#===========================
class NeuralNet2 (nn. Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super (NeuralNet2, self).__init__()
        self.linearl = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.linearl(x)
        out = self.relu(out)
        out = self.linear2(out)
        
        # sin softmax al final
        #===========================
        return out

#   Correr problema de múltiples clases
#===========================
model = NeuralNet2(input_size = 28*28, hidden_size = 5, num_classes = 3)
criterion = nn.CrossEntropyLoss()      # (aplica Softmax)