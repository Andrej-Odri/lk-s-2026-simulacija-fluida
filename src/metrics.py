import numpy as np

def CourantFriedrichLewy(brzina_x, brzina_y, dt, h):
    
    u_max = np.max(np.abs(brzina_x))
    v_max = np.max(np.abs(brzina_y))

    c_fl = (max(u_max, v_max) * dt) / h
    return c_fl

def DivergenceMetric(brzina_x, brzina_y):
    
    du_dx = brzina_x[:, 1:] - brzina_x[:, :-1]
    
    dv_dy = brzina_y[1:, :] - brzina_y[:-1, :]
    
    
    div = np.sum(np.abs(du_dx + dv_dy))
    return div

def Vorticity(brzina_x, brzina_y, h=1.0):


    du_dy = (brzina_x[1:, :] - brzina_x[:-1, :]) / h  
    dv_dx = (brzina_y[:, 1:] - brzina_y[:, :-1]) / h  

    target_rows = min(du_dy.shape[0], dv_dx.shape[0])
    target_cols = min(du_dy.shape[1], dv_dx.shape[1])
    
    
    start_row_u = (du_dy.shape[0] - target_rows) // 2
    start_col_u = (du_dy.shape[1] - target_cols) // 2
    
    start_row_v = (dv_dx.shape[0] - target_rows) // 2
    start_col_v = (dv_dx.shape[1] - target_cols) // 2
    
    
    du_dy_cropped = du_dy[start_row_u:start_row_u + target_rows, start_col_u:start_col_u + target_cols]
    dv_dx_cropped = dv_dx[start_row_v:start_row_v + target_rows, start_col_v:start_col_v + target_cols]
    
    
    rotacija = dv_dx_cropped - du_dy_cropped
    
    ukupni_vorticitet = np.sum(np.abs(rotacija))
    max_curl = np.max(np.abs(rotacija))
    
    return ukupni_vorticitet, max_curl

def KineticEnergy(brzina_x, brzina_y, rho):
    
    kineticka_energija = 0.5 * rho * (np.sum(brzina_x ** 2) + np.sum(brzina_y ** 2))
    return kineticka_energija