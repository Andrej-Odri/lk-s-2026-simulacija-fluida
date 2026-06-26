import numpy as np
from scipy.sparse.linalg import cg

def Univerzalna_Advekcija(polje, brzina_x, brzina_y, tip_pozicije, dt=0.05, h=0.1):
    novo_polje = np.zeros_like(polje)
    redova, kolone = polje.shape
    for i in range(redova):
        for j in range(kolone):
            if tip_pozicije == 'x_ivica':
                u_vetar = polje[i, j]
                i_top = max(0, i - 1)
                i_bot = min(brzina_y.shape[0] - 1, i)
                j_left = max(0, j - 1)
                j_right = min(brzina_y.shape[1] - 1, j)
                v_vetar = (brzina_y[i_top, j_left] + brzina_y[i_bot, j_left] + \
                           brzina_y[i_top, j_right] + brzina_y[i_bot, j_right]) / 4.0
            elif tip_pozicije == 'y_ivica':
                v_vetar = polje[i, j]
                i_top = max(0, i - 1)
                i_bot = min(brzina_x.shape[0] - 1, i)
                j_left = max(0, j - 1)
                j_right = min(brzina_x.shape[1] - 1, j)
                u_vetar = (brzina_x[i_top, j_left] + brzina_x[i_bot, j_left] + \
                           brzina_x[i_top, j_right] + brzina_x[i_bot, j_right]) / 4.0

            i_unazad = i - (v_vetar * dt / h)
            j_unazad = j - (u_vetar * dt / h)
            i_staro = max(0.0, min(float(redova - 1), i_unazad))
            j_staro = max(0.0, min(float(kolone - 1), j_unazad))
            i0 = int(np.floor(i_staro))
            i1 = min(redova - 1, i0 + 1)
            j0 = int(np.floor(j_staro))
            j1 = min(kolone - 1, j0 + 1)
            beta = i_staro - i0
            alfa = j_staro - j0
            novo_polje[i, j] = polje[i0, j0] * (1 - alfa) * (1 - beta) + \
                               polje[i0, j1] * alfa * (1 - beta) + \
                               polje[i1, j0] * (1 - alfa) * beta + \
                               polje[i1, j1] * alfa * beta
    return novo_polje

def Viskoznost(brzina_x, brzina_y, dt, h, ni):
    x_privremena = np.zeros_like(brzina_x)
    y_privremena = np.zeros_like(brzina_y)

    x_redovi, x_kolone = brzina_x.shape
    y_redovi, y_kolone = brzina_y.shape

    for i in range(1, x_redovi - 1):
        for j in range(1, x_kolone - 1):
            viskoznost_x = ni * dt * (brzina_x[i - 1, j] + brzina_x[i, j - 1] + brzina_x[i, j + 1] + \
                                      brzina_x[i + 1, j + 1] - 4*brzina_x[i, j]) / (h**2) 
            x_privremena[i, j] = brzina_x[i, j] + viskoznost_x

    for i in range(1, y_redovi - 1):
        for j in range(1, y_kolone - 1):
            viskoznost_y = ni * dt * (brzina_y[i - 1, j] + brzina_y[i, j - 1] + brzina_y[i, j + 1] + \
                                      brzina_y[i + 1, j + 1] - 4*brzina_y[i, j]) / (h**2) 
            y_privremena[i, j] = brzina_y[i, j] + viskoznost_y
    
    return x_privremena, y_privremena

def Univerzalna_Difuzija(polje, ni, dt, h):

    novo_polje = np.copy(polje) 
    redova, kolone = polje.shape
    
    for i in range(1, redova - 1):
        for j in range(1, kolone - 1):
            laplasijan = (polje[i-1, j] + polje[i+1, j] + \
                          polje[i, j-1] + polje[i, j+1] - 4.0 * polje[i, j]) / (h**2)
            
            novo_polje[i, j] = polje[i, j] + ni * dt * laplasijan
            
    return novo_polje

def IndexMap(tip_celije):
    
    N = len(tip_celije)

    mapa_indexa = np.zeros((N,N))
    cnt = 0
    for i in range(N):
        for j in range(N):
            if(tip_celije[i,j] == 0 or tip_celije[i,j] == 2):
                mapa_indexa[i,j] = -1
            else:
                mapa_indexa[i,j] = cnt
                cnt += 1
    
    return mapa_indexa
                
def Matrica_A(tip_celije):
    tip_celije = np.array(tip_celije)
    mapa_indexa = IndexMap(tip_celije)

    N = int(np.max(mapa_indexa) + 1)
    matrica_a = np.zeros((N, N)) 

    
    broj_redova, broj_kolona = mapa_indexa.shape

    
    for i in range(broj_redova):
        for j in range(broj_kolona):
            a = int(mapa_indexa[i, j])
            
            if a != -1:
                cnt = 0 
                
            
                if tip_celije[i-1, j] == 1:
                    cnt += 1
                    matrica_a[a, int(mapa_indexa[i-1, j])] = 1 
                if tip_celije[i+1, j] == 1:
                    cnt += 1
                    matrica_a[a, int(mapa_indexa[i+1, j])] = 1 
                if tip_celije[i, j-1] == 1:
                    cnt += 1
                    matrica_a[a, int(mapa_indexa[i, j-1])] = 1 
                if tip_celije[i, j+1] == 1:
                    cnt += 1
                    matrica_a[a, int(mapa_indexa[i, j+1])] = 1 

                
                if tip_celije[i-1, j] == 2:
                    cnt += 1
                if tip_celije[i+1, j] == 2:
                    cnt += 1
                if tip_celije[i, j-1] == 2:
                    cnt += 1
                if tip_celije[i, j+1] == 2:
                    cnt += 1

                matrica_a[a, a] = -cnt
                
    return matrica_a

def vectorB(tip_celije, brzina_x, brzina_y, rho, dt, h):
    tip_celije = np.array(tip_celije)
    mapa_indexa = IndexMap(tip_celije)
    N = int(np.max(mapa_indexa) + 1)
    b = np.zeros(N)
    broj_redova, broj_kolona = mapa_indexa.shape

    for i in range(broj_redova):
        for j in range(broj_kolona):
            if mapa_indexa[i,j] != -1:
                u_levo = brzina_x[i, j]
                u_desno = brzina_x[i, j+1]
                v_gore = brzina_y[i, j]
                v_dole = brzina_y[i+1, j]

                if tip_celije[i, j-1] == 0: u_levo = 0.0
                if tip_celije[i, j+1] == 0: u_desno = 0.0
                if tip_celije[i-1, j] == 0: v_gore = 0.0
                if tip_celije[i+1, j] == 0: v_dole = 0.0

                divegencija = (u_desno - u_levo) + (v_dole - v_gore)
                
                b[int(mapa_indexa[i,j])] = (rho * h / dt) * divegencija
    return b

def IzracunajPritisak(matrica_a, b_vektor, mapa_indexa, tip_celije, tol = 1e-5):
    
    P_vektor, _ = cg(matrica_a, b_vektor, rtol=tol)

    P_matrica = np.zeros(tip_celije.shape)

    broj_redova, broj_kolona = mapa_indexa.shape

    for i in range(broj_redova):
        for j in range(broj_kolona):
            idx = mapa_indexa[i, j]
            
            if idx != -1:
                P_matrica[i, j] = P_vektor[int(idx)]
                
    return P_matrica