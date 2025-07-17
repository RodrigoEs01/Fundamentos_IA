#===========================
#   Manejo de datos en Pytorch
#===========================
#   Rodrigo Espinosa Lopez
#   Fundamentos de IA
#   ESFM IPN 2025
#===========================

#   Módulos necesarios
#===========================
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#===========================
#   Bigdata debe dividirse en pequeños grupos de datos
#   
#   Ciclo de entrenamiento.-
#   for epoch in range(num_epochs):
#       #   Ciclo sobre todos los grupos de datos
#       #=========================
#       for i in range(total batches)
#
#   epoch : una evaluación y retropropagación para todos los datos de entrenamiento
#   total_batches : número total de subconjuntos de datos
#   batch_size : número de datos de entrenamiento en cada subconjunto
#   number_of_iterations : número de iteraciones sobre todos los datos de entrenamiento
#
#   e.g: 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch
#
#   DataLoader puede dividir los datos en grupos
#===========================

#===========================
#   Implementación de base de datos típica
#   implement init getitem , and len
#===========================

#   Hijo de Dataset
#===========================
class WineDataset (Dataset):
    def __init__(self):
        #   Inicializar, bajar datos, etc.
        #   Lectura con numpy o pandas
        #   Típicos datos separados por coma
        #   delimiter = símbolo delimitador
        #   skiprows = líneas de encabezado
        #===========================        
        xy = np.loadtxt('./data/wine/wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.n_samples = xy.shape[0]

        #   Primera columna es etiqueta de clase y el resto son características
        #===========================
        self.x_data = torch.from_numpy(xy[:, 1:])       # grupos del 1 en adelante
        self.y_data = torch.from_numpy(xy[:, [0]])      # grupo

    #   Permitir indexación para obtener el dato i de dataset [i]
    #   método getter
    #===========================

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    #   len(dataset) es el tamaño de la base de datos
    #===========================
    def __len__(self):
        return self.n_samples

#   Instanciar base de datos
#===========================    
dataset = WineDataset()

#   Leer características del primer dato
#===========================
first_data = dataset[0]
features, labels = first_data
print(features, labels)

#===========================
#   Cargar toda la base con DataLoader
#   reborujar los datos (shuffle) : bueno para el entrenamiento
#   num_workers : carga rápida utilizando múltiples procesos
#   SI COMETE UN ERROR EN LA CARGA, PONER num workers = 0
#===========================
train_loader = DataLoader(dataset=dataset,      #   Base de datos
                          batch_size = 4,       #   Cuatro grupos
                          shuffle = True,       #   Reborujados
                          num_workers = 2)      #   Dos procesos

#   Convertir en iterador y observar un dato al azar
#===========================
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print (features, labels)

#   Ciclo de aprendizaje vacío
#===========================
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        #   178 líneas, batch size 4, n iters=178/4-44.5 -> 45 iteraciones  
        #   Corre tu proceso de aprendizaje
        #   Diagnóstico
        #===========================
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {1+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
       
#   Algunas bases de datos existen en torchvision.datasets
#   e.g. MNIST, Fashion-MNIST, CIFAR10, COCO
#===========================
train_dataset = torchvision.datasets.MNIST(root='./data',
                                        train = True,
                                        transform = torchvision.transforms.ToTensor(),
                                        download = True)

train_loader = DataLoader(dataset = train_dataset,
                                        batch_size = 3,
                                        shuffle = True)

#   Look at one random sample
#===========================
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)