#===========================
#   Pytorch basico
#===========================
#   Rodrigo Espinosa Lopez
#   Fundamentos de IA
#   ESFM IPN 2025
#===========================

#   Modulos a importar
#===========================
import torch

#===========================
#   En Pytorch todo esta basado en operaciones
#   tensoriales. Un tensor vive en Rn x Rm x ... etc.
#===========================

#   Escalar vacio (trae basura)
#===========================
x = torch.empty(1)      #   escalar
print(x)

#   Vector vacio en R3
#===========================
x = torch.empty(3)
print(x)

#   Tensor vacio en R2 x R3
#===========================
x = torch.empty(2, 3)
print(x)

#   Tensor vacio en R2 x R2 x R3
#===========================
x = torch.empty(2, 2, 3)
print(x)

#===========================
#   torch.rand(size) :  numeros aleatorios [0, 1]
#===========================

#   Tensor de numeros aleatorios de R5 x R3
#===========================
x = torch.rand(5, 3)
print(x)

#===========================
#   torch.zeros(size) :  llenar con 0
#   torch.ones(size) :  llenar con 1
#===========================

#   Tensor de ceros de R5 x R3
#===========================
x = torch.zeros(5, 3)
print(x)

#   Tensor de unos de R5 x R3
#===========================
x = torch.ones(5, 3)
print(x)

#   Checar tamaño (Lista de dimensiones)
#===========================
print(x.size())

#   Checar tipo de datos (default es float32)
#===========================
print(x.dtype)

#   Especificando tipo de datos
#===========================
x = torch.zeros(5, 3, dtype = torch.float16)
print(x)
print(x.dtype)

#   Construir vector con datos
#===========================
x = torch.tensor([5.5, 3])
print(x.size())

#   Vector optimizable (variables del gradiente)
#===========================
x = torch.tensor([5.5, 3], requires_grad = True)

#   Suma de tensores
#===========================
y = torch.rand(2, 2)  
x = torch.rand(2, 2)
z = x + y
z = torch.add(x, y)
print(z)
y.add(x)
print(y)

#   Resta de tensores
#===========================
z = x - y
z = torch.sub(x, y)
print(z)

#   producto de tensores
#===========================
z = x * y
z = torch.mul(x, y)
print(z)

#   Divicion de tensores
#===========================
z = x / y
z = torch.div(x, y)
print(z)

#   Rebanadas de tensores
#===========================
x = torch.rand(5, 3)
print(x)
print(x[:, 0])      #   Todas las filas, columna 0
print(x[1, :])      #   fila 1, todas las columnas
print(x[1, 1])      #   elemento en (1, 1)

#   Valor del elemento en (1, 1)
#===========================
print(x[1, 1].item())

#   Cambiar forma con torch.view()
#===========================
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)       #   -1 : Se infiere de las otras dimensiones
#                           Si -1 Pytorch determinara automaticamente el tamaño necesario
print(x.size(), y.size(), z.size())

#   Convertir un tensor en arreglo de numpy y viceversa
#===========================
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))

#   Le suma 1 a todas las entradas
#===========================
a.add_(1)
print(a)
print(b)

#   De numpy atorch
#===========================
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

#   Le suma 1 a todas las entradas
#===========================
a += 1
print(a)
print(b)

#   De CPU a GPU (si hay CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")               #   Tarjeta de video con CUDA
    print("Tengo GPU " + str(device))
    y_d = torch.ones_like(x, device = device)   #   Crear tensor en el GPU
    x_d = x.to(device)                          #   Copiar a GPU o usar ''.to("cuda")''
    z_d = x_d + y_d
    
    #   z = z_d.numpy()     #   Numpy no maneja tensores en el GPU
    #                           de vuelta al CPU
    z = z_d.to("cpu")
    z = z.numpy()
    print(z)