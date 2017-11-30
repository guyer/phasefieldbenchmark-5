from scipy.optimize import fsolve
import fipy as fp

def mesh_and_boundaries(params):
    """Generate a 2D grid appropriate for the parameters

    """

    dx = dy = params["cellSize"]
    Nx = int(params["Lx"] / dx)
    Ny = int(params["Ly"] / dx)
    dx_variable = [dx] * Nx
    
    dy_variable = [dy] * Ny

    mesh = fp.Grid2D(dx=dx_variable, dy=dy_variable)

    X, Y = mesh.faceCenters

    print dx_variable
    
    inlet = mesh.facesLeft
    outlet = mesh.facesRight
    walls = mesh.facesTop | mesh.facesBottom
    top_right = outlet & (Y > params["Ly"] - dy)

    return mesh, inlet, outlet, walls, top_right