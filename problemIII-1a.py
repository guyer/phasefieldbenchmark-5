import os
import sys
import argparse
import time
import uuid 

import numpy as np

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

parser = argparse.ArgumentParser()
parser.add_argument("--output", help="directory to store results in",
                    default=str(uuid.uuid4()))
parser.add_argument("--sweeps", help="number of nonlinear sweeps to take",
                    type=int, default=10)
args, unknowns = parser.parse_known_args()
                    
if parallelComm.procID == 0:
    print "storing results in {0}".format(args.output)
    data = dtr.Treant(args.output)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
data.categories['args'] = " ".join(sys.argv)
data.categories['sweeps'] = args.sweeps
data.categories['commit'] = os.popen('git log --pretty="%H" -1').read().strip()
data.categories['diff'] = os.popen('git diff').read()
    
L = 1.0
N = 50
dL = L / N
viscosity = 1
U = 1.
pressureRelaxation = 0.8
velocityRelaxation = 0.5



mesh = fp.Grid2D(nx=N, ny=N, dx=dL, dy=dL)
volumes = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)

pressure = fp.CellVariable(mesh=mesh, name="$p$")
pressureCorrection = fp.CellVariable(mesh=mesh, name="$p'$")
xVelocity = fp.CellVariable(mesh=mesh, name="$u_x$")
yVelocity = fp.CellVariable(mesh=mesh, name="$u_y$")

velocity = fp.FaceVariable(mesh=mesh, name=r"$\vec{u}$", rank=1)

xVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.,0.])
yVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.,1.])

ap = fp.CellVariable(mesh=mesh, value=1.)
coeff = 1./ ap.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = fp.DiffusionTerm(coeff=coeff) - velocity.divergence

contrvolume = volumes.arithmeticFaceValue

x, y = mesh.cellCenters
X, Y = mesh.faceCenters

xVelocity.constrain(0., mesh.facesRight | mesh.facesLeft | mesh.facesBottom)
xVelocity.constrain(U, mesh.facesTop)
yVelocity.constrain(0., mesh.exteriorFaces)
X, Y = mesh.faceCenters
pressureCorrection.constrain(0., mesh.facesLeft & (Y < dL))

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
    velocity[0, mesh.facesTop.value] = U

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## left bottom point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection )
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               ap * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               ap * mesh.cellVolumes)

    fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                        filename=data["sweep={}.tar.gz".format(sweep)].make().abspath)
                        
    print 'sweep:',sweep,', x residual:',xres, \
                         ', y residual',yres, \
                         ', p residual:',pres, \
                         ', continuity:',max(abs(rhs))

                            
data.categories['elapsed'] = time.clock() - start
