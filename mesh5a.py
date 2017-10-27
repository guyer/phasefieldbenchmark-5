from scipy.optimize import fsolve
import fipy as fp

def mesh_and_boundaries(Lx, Ly, dx, dy, compression):
    """Generate a 2D grid appropriate for the parameters

    """
    def fn(f, N):
        '''Root solving kernel for compression factor
        
        Determine f(N), such that $\Delta x \sum_{i=0}^N f^i = 2 \Delta x$
        '''
        return (1 - f**N) / (1 - f) - 2.
        
    N = 1 + compression
    compression = fsolve(fn, x0=[.5], args=(N))[0]

    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    dx_variable = [dx] * (Nx - 2) + [dx * compression**i for i in range(N+1)]

    dy_variable = [dy] * Ny

    mesh = fp.Grid2D(dx=dx_variable, dy=dy_variable)

    X, Y = mesh.faceCenters

    inlet = mesh.facesLeft
    outlet = mesh.facesRight
    walls = mesh.facesTop | mesh.facesBottom
    top_right = outlet & (Y > Ly - dy)

    return mesh, inlet, outlet, walls, top_right