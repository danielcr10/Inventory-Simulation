#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats
from ipywidgets import FloatProgress
from IPython.display import display

# %%capture
# from tqdm import tqdm_notebook as tqdm
# from tqdm.notebook import tqdm
from tqdm import tqdm
# tqdm().pandas()

# get_ipython().run_line_magic('matplotlib', 'inline')

T = 6.0
maxEstoque = 15 # máximo disponível
minEstoque = 5 # mínimo disponível
x0 = 3 # quantidade inicial no inventário
precoPorUnidade = 50 # preço por unidade
und = 2 # unidade de tempo antes do pedido chegar
custo = 2 # custo de manutenção do inventário
maxRate = 10 # número máximo de clientes por unidade de tempo
recebeAteFechar = True # pedidos podem chegar depois de fechar
custoAteFechar = True # o custo de manutenção é cobrado até o fechamento da loja
custoPorUnidade = 10

def cost(x): 
    return x * custoPorUnidade

def rate(t):
    if (t < 1.0):
        return 10.0
    elif (t < 2.0):
        return 5.0
    elif (t < 3.0):
        return 10.0
    elif (t < 4.0):
        return 2.0
    elif (t < 5.0):
        return 5.0
    else:
        return 10.0

def proxChegada(t,rate,maxRate):
    while(1):
        Z = exponencial(1,maxRate)[0]
        t = t + Z
        U = np.random.sample(1)
        if U < rate(t) / maxRate:
            return t

def exponencial(nsamples,rate):
    x = np.zeros(nsamples)
    u = np.random.sample(nsamples)
    for i in range(nsamples):
        x[i] = - math.log(1.0 - u[i]) / rate
    return x


def inventario(G,T,maxEstoque,minEstoque,cost,x0,precoPorUnidade,und,custo,rate,maxRate,recebeAteFechar,custoAteFechar):
    
    qtdCusto = 0.0 # valor de custos de pedidos
    qtdReceita = 0.0 # valor da receita obtida
    valCusto = 0.0 # valor total do custo de manutenção
    qtdInventario = x0 # quantidade no inventário
    qtdPedido = 0 # quantidade no pedido
    t = 0.0 
    t0 = proxChegada(t,rate,maxRate) # hora de chegada do próximo cliente
    t1 = 1.0e+30 # tempo de entrega do pedido
    T0 = []
    T1 = []
    
    while(1):
        
        #cliente chegou depois da entrega
        if t0 < t1 and t0 <= T:
            valCusto = valCusto + (t0 - t) * qtdInventario * custo
            t = t0
            T0.append(t0)
            demanda = G(1,rate(t))[0] # demanda do cliente
            valorPedidoPreenchido = min(demanda,qtdInventario) # valor do pedido que pode ser vendido
            qtdReceita = qtdReceita + valorPedidoPreenchido * precoPorUnidade
            qtdInventario = qtdInventario - valorPedidoPreenchido
            # política de pedidos
            if qtdInventario < minEstoque and qtdPedido == 0:
                qtdPedido = maxEstoque - qtdInventario
                t1 = t + und
            t0 = proxChegada(t,rate,maxRate)
            
        # pedido chega antes do próximo cliente
        elif (t1 <= t0 or t0 > T) and t1 <= T:
            valCusto = valCusto + (t1 - t) * custo * custo # Porque aqui é custo * custo?
            t = t1
            T1.append(t1)
            qtdCusto = qtdCusto + cost(qtdPedido)
            qtdInventario = qtdInventario + qtdPedido
            qtdPedido = 0
            t1 = 1.0e+30
            
        else:
            # loja é fechada e ainda existe um pedido pendente
            if qtdPedido > 0 and recebeAteFechar:
                valCusto = valCusto + (t1 - t) * qtdInventario * custo
                t = t1
                T1.append(t1)
                qtdCusto = qtdCusto + cost(qtdPedido)
                qtdInventario = qtdInventario + qtdPedido
                qtdPedido = 0
                
            elif custoAteFechar:
                valCusto = valCusto + (T - t) * qtdInventario * custo
                
            lucroTotal = qtdReceita - qtdCusto - valCusto
            lucroPorUnidade = lucroTotal / T # Lucro por unidade de tempo
            return qtdReceita, lucroTotal, lucroPorUnidade, T0, T1


qtdReceita, lucroTotal, lucroPorUnidade, T0, T1 = inventario(exponencial,T,maxEstoque,minEstoque,cost,x0,precoPorUnidade,und,custo,rate,maxRate,recebeAteFechar,custoAteFechar)
print("Lucro Total: " + str(lucroTotal) + " = "+str(round(lucroTotal/qtdReceita*100, 2))+"%")
print("Lucro por unidade de tempo: " + str(lucroPorUnidade))
print()
print("Horários de chegada dos clientes: " + str(T0))
print()
print("Horário de chegada dos pedidos: " + str(T1))
print("\n\n")


def toleranciaDoInventario(tol,alpha):
    x = np.zeros(100)
    for i in range(100):
        qtdReceita, lucroTotal,lucroPorUnidade,T0,T1 = inventario(exponencial,T,maxEstoque,minEstoque,cost,x0,precoPorUnidade,und,custo,rate,maxRate,recebeAteFechar,custoAteFechar)
        x[i] = lucroPorUnidade
    n  = 100
    m  = np.mean(x)
    s2 = np.var(x)
    zab2 = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
    while(2.0 * (s2 / n) * zab2 * zab2 > tol**2):
        qtdReceita, lucroTotal,nx,T0,T1 = inventario(exponencial,T,maxEstoque,minEstoque,cost,x0,precoPorUnidade,und,custo,rate,maxRate,recebeAteFechar,custoAteFechar)
        nm  = m + (nx - m) / (n + 1)
        ns2 = (1.0 - 1.0 / n) * s2 + (n + 1.0) * (nm - m)**2
        n = n + 1
        m  = nm
        s2 = ns2 
        x = np.append(x,nx)
    return m,s2,n,x

tol = 0.1
alpha = 0.05

m,s2,n,x = toleranciaDoInventario(tol,alpha)
print("Lucro médio por unidade de tempo: " + str(m))
print(s2)
print("Número de cenários necessários: " + str(n))
zab2 = scipy.stats.norm.ppf(1.0 - alpha / 2.0)
print("A média está no intervalo [" + str(m - math.sqrt(s2 / n) * zab2) + "," + str(m + math.sqrt(s2 / n) * zab2)       + "] com a probabilidade de " + str(1.0 - alpha) + ".")
print("\n\n")


plt.hist(x, 20, density = 1, facecolor ='green', alpha = 0.5)
plt.title("Distribuição de lucro por unidade de tempo")
plt.show()

lucro_maximo = -1.0e+30
best_maxEstoque = 0
best_minEstoque = 0

f1 = FloatProgress(min=0, max=11)
f2 = FloatProgress(min=0, max=11)
display(f1)
display(f2)

# rMaxEstoque = range(15,26)
# rMinEstoque = (15,4,-1)

# with tqdm(total=len(rMaxEstoque)) as pbar:
for maxEstoque in tqdm(range(15,26)):
    for minEstoque in tqdm(range(15,4,-1)):
        m,s2,n,x = toleranciaDoInventario(tol,alpha)
        if m > lucro_maximo:
            lucro_maximo = m
            best_maxEstoque = maxEstoque
            best_minEstoque = minEstoque
        f2.value = f2.value + 1
    f1.value = f1.value + 1
            
print("Lucro máximo por unidade de tempo: " + str(lucro_maximo))
print("Melhor política de pedidos (minEstoque, maxEstoque): (" + str(best_minEstoque) + "," + str(best_maxEstoque) + ")")