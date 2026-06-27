import numpy as np
from scipy.sparse.linalg import cg

def Univerzalna_Advekcija(polje, brzina_x, brzina_y, tip_pozicije, dt=0.05, h=0.1):
    redova, kolone = polje.shape
    i_grid, j_grid = np.indices(polje.shape)

    if tip_pozicije == 'x_ivica':
        u_vetar = polje
        i_top = np.clip(i_grid - 1, 0, brzina_y.shape[0] - 1)
        i_bot = np.clip(i_grid, 0, brzina_y.shape[0] - 1)
        j_left = np.clip(j_grid - 1, 0, brzina_y.shape[1] - 1)
        j_right = np.clip(j_grid, 0, brzina_y.shape[1] - 1)
        v_vetar = (
            brzina_y[i_top, j_left]
            + brzina_y[i_bot, j_left]
            + brzina_y[i_top, j_right]
            + brzina_y[i_bot, j_right]
        ) / 4.0
    elif tip_pozicije == 'y_ivica':
        v_vetar = polje
        i_top = np.clip(i_grid - 1, 0, brzina_x.shape[0] - 1)
        i_bot = np.clip(i_grid, 0, brzina_x.shape[0] - 1)
        j_left = np.clip(j_grid - 1, 0, brzina_x.shape[1] - 1)
        j_right = np.clip(j_grid, 0, brzina_x.shape[1] - 1)
        u_vetar = (
            brzina_x[i_top, j_left]
            + brzina_x[i_bot, j_left]
            + brzina_x[i_top, j_right]
            + brzina_x[i_bot, j_right]
        ) / 4.0
    else:
        raise ValueError(f"Nepoznat tip pozicije: {tip_pozicije}")

    i_staro = np.clip(i_grid - (v_vetar * dt / h), 0.0, float(redova - 1))
    j_staro = np.clip(j_grid - (u_vetar * dt / h), 0.0, float(kolone - 1))

    i0 = np.floor(i_staro).astype(int)
    j0 = np.floor(j_staro).astype(int)
    i1 = np.minimum(redova - 1, i0 + 1)
    j1 = np.minimum(kolone - 1, j0 + 1)

    beta = i_staro - i0
    alfa = j_staro - j0

    return (
        polje[i0, j0] * (1 - alfa) * (1 - beta)
        + polje[i0, j1] * alfa * (1 - beta)
        + polje[i1, j0] * (1 - alfa) * beta
        + polje[i1, j1] * alfa * beta
    )

def Viskoznost(brzina_x, brzina_y, dt, h, ni):
    x_privremena = np.zeros_like(brzina_x)
    y_privremena = np.zeros_like(brzina_y)

    x_privremena[1:-1, 1:-1] = brzina_x[1:-1, 1:-1] + ni * dt * (
        brzina_x[:-2, 1:-1]
        + brzina_x[1:-1, :-2]
        + brzina_x[1:-1, 2:]
        + brzina_x[2:, 2:]
        - 4 * brzina_x[1:-1, 1:-1]
    ) / (h**2)

    y_privremena[1:-1, 1:-1] = brzina_y[1:-1, 1:-1] + ni * dt * (
        brzina_y[:-2, 1:-1]
        + brzina_y[1:-1, :-2]
        + brzina_y[1:-1, 2:]
        + brzina_y[2:, 2:]
        - 4 * brzina_y[1:-1, 1:-1]
    ) / (h**2)
    
    return x_privremena, y_privremena

def Univerzalna_Difuzija(polje, ni, dt, h):

    novo_polje = np.copy(polje) 
    laplasijan = (
        polje[:-2, 1:-1]
        + polje[2:, 1:-1]
        + polje[1:-1, :-2]
        + polje[1:-1, 2:]
        - 4.0 * polje[1:-1, 1:-1]
    ) / (h**2)
    novo_polje[1:-1, 1:-1] = polje[1:-1, 1:-1] + ni * dt * laplasijan
            
    return novo_polje

def IndexMap(tip_celije):
    tip_celije = np.asarray(tip_celije)
    mapa_indexa = -np.ones(tip_celije.shape, dtype=int)
    fluid_mask = (tip_celije != 0) & (tip_celije != 2)
    mapa_indexa[fluid_mask] = np.arange(np.count_nonzero(fluid_mask))
    
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

    u_levo = brzina_x[:, :-1].copy()
    u_desno = brzina_x[:, 1:].copy()
    v_gore = brzina_y[:-1, :].copy()
    v_dole = brzina_y[1:, :].copy()

    u_levo[:, 1:] = np.where(tip_celije[:, :-1] == 0, 0.0, u_levo[:, 1:])
    u_desno[:, :-1] = np.where(tip_celije[:, 1:] == 0, 0.0, u_desno[:, :-1])
    v_gore[1:, :] = np.where(tip_celije[:-1, :] == 0, 0.0, v_gore[1:, :])
    v_dole[:-1, :] = np.where(tip_celije[1:, :] == 0, 0.0, v_dole[:-1, :])

    divergencija = (u_desno - u_levo) + (v_dole - v_gore)
    fluid_mask = mapa_indexa != -1
    b[mapa_indexa[fluid_mask]] = (rho * h / dt) * divergencija[fluid_mask]
    return b

def IzracunajPritisak(matrica_a, b_vektor, mapa_indexa, tip_celije, tol = 1e-5):
    
    P_vektor, _ = cg(matrica_a, b_vektor, rtol=tol)

    P_matrica = np.zeros(tip_celije.shape)
    fluid_mask = mapa_indexa != -1
    P_matrica[fluid_mask] = P_vektor[mapa_indexa[fluid_mask].astype(int)]
                
    return P_matrica

def add_vortex(brzina_x, brzina_y, cx, cy, radius, strength, h):

    eps = 1e-8

    i_x, j_x = np.indices(brzina_x.shape)
    dx = j_x * h - cx
    dy = (i_x + 0.5) * h - cy
    r = np.sqrt(dx * dx + dy * dy)
    mask = r < radius
    factor = np.zeros_like(brzina_x, dtype=float)
    factor[mask] = (1.0 - np.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
    brzina_x[mask] += -strength * dy[mask] * factor[mask]

    i_y, j_y = np.indices(brzina_y.shape)
    dx = (j_y + 0.5) * h - cx
    dy = i_y * h - cy
    r = np.sqrt(dx * dx + dy * dy)
    mask = r < radius
    factor = np.zeros_like(brzina_y, dtype=float)
    factor[mask] = (1.0 - np.exp(-(r[mask] / radius) ** 2)) / (r[mask] + eps)
    brzina_y[mask] += strength * dx[mask] * factor[mask]

def initialize_blob(brzina_x, brzina_y, N):

    brzina_x.fill(0)
    brzina_y.fill(0)

    centar_i = N // 2
    centar_j = N // 4
    radijus = 5

    i_x, j_x = np.indices(brzina_x.shape)
    brzina_x[(i_x - centar_i) ** 2 + (j_x - centar_j) ** 2 < radijus**2] = 10

    i_y, j_y = np.indices(brzina_y.shape)
    brzina_y[(i_y - centar_i) ** 2 + (j_y - centar_j) ** 2 < radijus**2] = 10

def initialize_single_vortex(brzina_x, brzina_y, N, h):

    brzina_x.fill(0)
    brzina_y.fill(0)

    add_vortex(
        brzina_x,
        brzina_y,
        N*h*0.5,
        N*h*0.5,
        0.8,
        40,
        h
    )   

def initialize_double_vortex(brzina_x, brzina_y, N, h):

    brzina_x.fill(0)
    brzina_y.fill(0)

    add_vortex(brzina_x, brzina_y,
               N*h*0.30,
               N*h*0.50,
               0.6,
               40,
               h)

    add_vortex(brzina_x, brzina_y,
               N*h*0.70,
               N*h*0.50,
               0.6,
               -40,
               h)

def initialize_four_vortices(brzina_x, brzina_y, N, h):

    brzina_x.fill(0)
    brzina_y.fill(0)

    vortices = [
        (0.30,0.30,40),
        (0.70,0.30,-40),
        (0.30,0.70,-40),
        (0.70,0.70,40)
    ]

    for px,py,s in vortices:
        add_vortex(
            brzina_x,
            brzina_y,
            N*h*px,
            N*h*py,
            0.45,
            s,
            h
        )

def initialize_shear_layer(brzina_x, brzina_y, strength, N):

    brzina_x.fill(0)
    brzina_y.fill(0)

    brzina_x[: N // 2, :] = strength
    brzina_x[N // 2 :, :] = -strength

def initialize_taylor_green(brzina_x, brzina_y, N, h):

    brzina_x.fill(0)
    brzina_y.fill(0)

    U0 = 20
    L = N*h

    i_x, j_x = np.indices(brzina_x.shape)
    x = j_x * h
    y = (i_x + 0.5) * h
    brzina_x[:, :] = U0 * np.sin(2 * np.pi * x / L) * np.cos(2 * np.pi * y / L)

    i_y, j_y = np.indices(brzina_y.shape)
    x = (j_y + 0.5) * h
    y = i_y * h
    brzina_y[:, :] = -U0 * np.cos(2 * np.pi * x / L) * np.sin(2 * np.pi * y / L)
