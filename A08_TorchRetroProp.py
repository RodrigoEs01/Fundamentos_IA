#===========================
#   Gradientes simples con pytorch
#===========================
#   Rodrigo Espinosa Lopez
#   Fundamentos de IA
#   ESFM IPN 2025
#===========================

#   Modulos a importar
#===========================
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#   Variable de diferenciacion (d/dw) -> requires_grad = true
#===========================
w = torch.tensor(1.0, requires_grad = True)

#   Evaluacion calculo de costo
#===========================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#   Nuevos coeficientes (descenso del gradiente)
#   repetir evaluacion y retropropagacion
#===========================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
print(w)