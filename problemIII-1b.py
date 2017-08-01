import os
import sys
import argparse
import time
import uuid 

from scipy.optimize import fsolve

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="directory to store results in",
                    default=os.path.join("Data", str(uuid.uuid4())))
parser.add_argument("--sweeps", help="number of nonlinear sweeps to take",
                    type=int, default=10)
parser.add_argument("--check", help="period of sweeps to checkpoint data",
                    type=int, default=1)
parser.add_argument("--cellSize", help="typlical cell dimension",
                    type=float, default=1.0)
args, unknowns = parser.parse_known_args()
                    
if parallelComm.procID == 0:
    print "storing results in {0}".format(args.output)
    data = dtr.Treant(args.output)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
data.categories['problem'] = "III-1b"
data.categories['args'] = " ".join(sys.argv)
data.categories['sweeps'] = args.sweeps
data.categories['cellSize'] = args.cellSize
data.categories['commit'] = os.popen('git log --pretty="%H" -1').read().strip()
data.categories['diff'] = os.popen('git diff').read()
    
viscosity = 1
density = 100.
gravity = [0., -0.001]
pressureRelaxation = 0.8
velocityRelaxation = 0.5

mesh = fp.Gmsh2D('''
cellSize = %(cellSize)g;
                 
Point(1) = {0, 0, 0, cellSize};
Point(2) = {30, 0, 0, cellSize};
Point(3) = {30, 6, 0, cellSize};
Point(4) = {0, 6, 0, cellSize};
Point(5) = {7, 2.5, 0, cellSize};
Point(6) = {7, 4, 0, cellSize};
Point(7) = {6, 2.5, 0, cellSize};
Point(8) = {7, 1, 0, cellSize};
Point(9) = {8, 2.5, 0, cellSize};
                 
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Ellipse(5) = {7, 5, 6, 6};
Ellipse(6) = {6, 5, 9, 9};
Ellipse(7) = {9, 5, 8, 8};
Ellipse(8) = {8, 5, 7, 7};

Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};
Plane Surface(11) = {9, 10};

Physical Surface("cells") = {11};
                 
Physical Line("bottom") = {1};
Physical Line("right") = {2};
Physical Line("top") = {3};
Physical Line("left") = {4};
Physical Line("hole") = {5, 6, 7, 8};
''' % dict(cellSize=args.cellSize))

volumes = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)

pressure = fp.CellVariable(mesh=mesh, name="$p$")
pressureCorrection = fp.CellVariable(mesh=mesh, name="$p'$")
xVelocity = fp.CellVariable(mesh=mesh, name="$u_x$")
yVelocity = fp.CellVariable(mesh=mesh, name="$u_y$")

velocity = fp.FaceVariable(mesh=mesh, name=r"$\vec{u}$", rank=1)

xVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.,0.]) - density * gravity[0]
yVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.,1.]) - density * gravity[1]

ap = fp.CellVariable(mesh=mesh, value=1.)
coeff = 1./ ap.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = fp.DiffusionTerm(coeff=coeff) - velocity.divergence

contrvolume = volumes.arithmeticFaceValue

x, y = mesh.cellCenters
X, Y = mesh.faceCenters

def inlet(yy):
    return -0.001 * (yy - 3)**2 + 0.009
    
xVelocity.constrain(inlet(Y), mesh.physicalFaces["left"])
xVelocity.constrain(0., (mesh.physicalFaces["top"] 
                         | mesh.physicalFaces["bottom"] 
                         | mesh.physicalFaces["hole"]))
                         
yVelocity.constrain(0., (mesh.physicalFaces["top"] 
                         | mesh.physicalFaces["bottom"]
                         | mesh.physicalFaces["left"]
                         | mesh.physicalFaces["hole"]))

pressureCorrection.constrain(0., mesh.physicalFaces["right"] & (Y > max(Y) - args.cellSize))
# pressureCorrection.constrain(0., mesh.physicalFaces["right"])

with open(data['residuals.npy'].make().abspath, 'a') as f:
    f.write("{}\t{}\t{}\t{}\t{}\n".format("sweep", "x_residual", "y_residual", "p_residual", "continuity"))

start = time.clock()

for sweep in range(args.sweeps):
    ## solve the Stokes equations to get starred values
    xVelocityEq.cacheMatrix()
    xres = xVelocityEq.sweep(var=xVelocity,
                             underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix

    yres = yVelocityEq.sweep(var=yVelocity,
                             underRelaxation=velocityRelaxation)

    ## update the ap coefficient from the matrix diagonal
    ap[:] = -xmat.takeDiagonal()

    ## update the face velocities based on starred values with the
    ## Rhie-Chow correction.
    ## cell pressure gradient
    presgrad = pressure.grad
    ## face pressure gradient
    facepresgrad = pressure.faceGrad

    velocity[0] = xVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[0].arithmeticFaceValue-facepresgrad[0])
    velocity[1] = yVelocity.arithmeticFaceValue \
         + contrvolume / ap.arithmeticFaceValue * \
           (presgrad[1].arithmeticFaceValue-facepresgrad[1])
    velocity[..., mesh.exteriorFaces.value] = 0.
    velocity[0, mesh.physicalFaces["left"].value] = inlet(Y)[mesh.physicalFaces["left"].value]
    velocity[0, mesh.physicalFaces["right"].value] = xVelocity.faceValue[mesh.physicalFaces["right"].value]
    velocity[1, mesh.physicalFaces["right"].value] = yVelocity.faceValue[mesh.physicalFaces["right"].value]

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection)
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               ap * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               ap * mesh.cellVolumes)

    if sweep % args.check == 0:
        fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                            filename=data["sweep={}.tar.gz".format(sweep)].make().abspath)
                   
    with open(data['residuals.npy'].make().abspath, 'a') as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format(sweep, xres, yres, pres, max(abs(rhs))))
                            
data.categories['elapsed'] = time.clock() - start
