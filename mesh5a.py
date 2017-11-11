from scipy.optimize import fsolve
import fipy as fp

def mesh_and_boundaries(params):
    """Generate a 2D grid appropriate for the parameters

    """
    def fn(f, N):
        '''Root solving kernel for compression factor
        
        Determine f(N), such that $\Delta x \sum_{i=0}^N f^i = 2 \Delta x$
        '''
        return (1 - f**N) / (1 - f) - 2.
        
    N = 1 + params["compression"]
    compression = fsolve(fn, x0=[.5], args=(N))[0]

    dx = dy = params["cellSize"]
    Nx = int(params["Lx"] / dx)
    Ny = int(params["Ly"] / dx)
    dx_variable = [dx] * (Nx - N) + [dx * params["compression"]**i for i in range(N)]
    
    dy_variable = [dy] * Ny

    mesh = fp.Grid2D(dx=dx_variable, dy=dy_variable)

    X, Y = mesh.faceCenters

    inlet = mesh.facesLeft
    outlet = mesh.facesRight
    walls = mesh.facesTop | mesh.facesBottom
    top_right = outlet & (Y > params["Ly"] - dy)

    return mesh, inlet, outlet, walls, top_right