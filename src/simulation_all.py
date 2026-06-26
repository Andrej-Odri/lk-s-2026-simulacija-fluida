import numpy as np
import matplotlib.pyplot as plt
from essential_functions import Univerzalna_Advekcija, Univerzalna_Difuzija, IndexMap, vectorB, Matrica_A, IzracunajPritisak
import matplotlib.animation as animation
from scipy.sparse.linalg import cg

# ==========================================================
# GLAVNA INICIJALIZACIJA SIMULACIJE (32x32)
# ==========================================================
N = 32
dt = 0.01 
h = 0.1
ni = 0.1    
rho = 1.0

# 1. Kreiranje mrežice (1=FLUID, 0=SOLID okviri na ivicama)
tip_celije = np.ones((N, N), dtype=int)
tip_celije[0, :] = 0   
tip_celije[-1, :] = 0  
tip_celije[:, 0] = 0   
tip_celije[:, -1] = 0  

# 2. Priprema matrice sistema
mapa_indexa = IndexMap(tip_celije)
A_sistem = Matrica_A(tip_celije) 

# 3. Inicijalizacija Staggered Grid brzina
brzina_x = np.zeros((N, N+1))
brzina_y = np.zeros((N+1, N))

# Početni mehur brzine
centar_i, centar_j = N // 2, N // 4
radijus = 5

for i in range(brzina_x.shape[0]):
    for j in range(brzina_x.shape[1]):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
            brzina_x[i, j] = 10.0

for i in range(brzina_y.shape[0]):
    for j in range(brzina_y.shape[1]):
        if (i - centar_i)**2 + (j - centar_j)**2 < radijus**2:
            brzina_y[i, j] = 10.0

# Koordinate za Quiver
X, Y = np.meshgrid(np.arange(N), np.arange(N))

fig, ax = plt.subplots(figsize=(8, 8))

P_prikaz = np.zeros((N, N))
im = ax.imshow(P_prikaz, cmap='jet', origin='upper', extent=[-0.5, N-0.5, N-0.5, -0.5])
kviver = ax.quiver(X, Y, np.zeros((N, N)), np.zeros((N, N)), color='white', scale=100, width=0.003)

ax.set_xlim(-0.5, N-0.5)
ax.set_ylim(N-0.5, -0.5)
title = ax.set_title("Fluid Simulacija | Pritisak i Brzina | Frejm 0")
fig.colorbar(im, ax=ax, label="Pritisak (P)")

# ==========================================================
# GLAVNA PETLJA ANIMACIJE
# ==========================================================
def update(frejm):
    global brzina_x, brzina_y
    
    # KORAK 1: Vektor B preko tvoje funkcije
    b_vektor = vectorB(tip_celije, brzina_x, brzina_y, rho, dt, h)
    
    # KORAK 2: Poziv TVOJE funkcije za računanje 2D matrice pritiska
    P_matrica = IzracunajPritisak(A_sistem, b_vektor, mapa_indexa, tip_celije, tol=1e-5)
                
    # KORAK 3: Korekcija brzina (Pressure Projection)
    # U update(frejm) funkciji promeni minuse u pluseve:

    # KORAK 3: Korekcija brzina (Zamenjeni minusi u pluseve zbog usaglašavanja znaka pritiska)
    # KORAK 3: Korekcija brzina (Vraćamo minuse jer smo sredili znak u vectorB)
    # KORAK 3: Korekcija brzina (Pressure Projection)
    for i in range(N):
        for j in range(1, N): # brzina_x je velicine (N, N+1)
            # Brzina se menja samo ako su obe susedne celije koje dele ivicu fluid
            if tip_celije[i, j] != 0 and tip_celije[i, j-1] != 0:
                brzina_x[i, j] = brzina_x[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i, j-1])
            else:
                brzina_x[i, j] = 0.0 # ODRŽAVAMO ZID TVRDIM
            
    for i in range(1, N): # brzina_y je velicine (N+1, N)
        for j in range(N):
            if tip_celije[i, j] != 0 and tip_celije[i-1, j] != 0:
                brzina_y[i, j] = brzina_y[i, j] - (dt / (rho * h)) * (P_matrica[i, j] - P_matrica[i-1, j])
            else:
                brzina_y[i, j] = 0.0
    # KORAK 4: Advekcija
    bx_advektovano = Univerzalna_Advekcija(brzina_x, brzina_x, brzina_y, 'x_ivica', dt, h)
    by_advektovano = Univerzalna_Advekcija(brzina_y, brzina_x, brzina_y, 'y_ivica', dt, h)
    
    # KORAK 5: Difuzija
    brzina_x = Univerzalna_Difuzija(bx_advektovano, ni, dt, h)
    brzina_y = Univerzalna_Difuzija(by_advektovano, ni, dt, h)
    
    # Srednje brzine za strelice
    U_centar = (brzina_x[:, :-1] + brzina_x[:, 1:]) / 2.0
    V_centar = (brzina_y[:-1, :] + brzina_y[1:, :]) / 2.0
    
    # Osvežavanje ekrana
    im.set_array(P_matrica)
    im.set_clim(vmin=np.min(P_matrica), vmax=np.max(P_matrica)) 
    kviver.set_UVC(U_centar, -V_centar)
    title.set_text(f"Fluid Simulacija | Pritisak i Brzina | Frejm {frejm}")
    
    return im, kviver, title

ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=False, repeat=False)
##plt.show()

# --- ČUVANJE U GIF ---
print("Učitavam frejmove i pravim GIF... (ovo može potrajati malo)")

# Kreiramo writer objekat (fps=30 znači 30 frejmova u sekundi)
writer = animation.PillowWriter(fps=30)

# Čuvamo animaciju pod imenom 'simulacija_fluida.gif'
ani.save('simulacija_fluida.gif', writer=writer)

print("GIF uspešno sačuvan kao 'simulacija_fluida.gif'!")