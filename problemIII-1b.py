import os
import sys
import yaml

from scipy.optimize import fsolve

import datreant.core as dtr

import fipy as fp
from fipy.tools.numerix import cos, sin
from fipy.tools import parallelComm

yamlfile = sys.argv[1]

with open(yamlfile, 'r') as f:
    params = yaml.load(f)

try:
    from sumatra.projects import load_project
    project = load_project(os.getcwd())
    record = project.get_record(params["sumatra_label"])
    output = record.datastore.root
except:
    # either there's no sumatra, no sumatra project, or no sumatra_label
    # this will be the case if this script is run directly
    output = os.getcwd()

if parallelComm.procID == 0:
    print "storing results in {0}".format(output)
    data = dtr.Treant(output)
else:
    class dummyTreant(object):
        categories = dict()
        
    data = dummyTreant()
    
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
''' % dict(cellSize=params["cellSize"]))

inlet = mesh.physicalFaces["left"]
outlet = mesh.physicalFaces["right"]
walls = mesh.physicalFaces["top"] | mesh.physicalFaces["bottom"] | mesh.physicalFaces["hole"]
top_right = outlet & (Y > max(Y) - args.cellSize)

volumes = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)

pressure = fp.CellVariable(mesh=mesh, name="$p$")
pressureCorrection = fp.CellVariable(mesh=mesh, name="$p'$")
xVelocity = fp.CellVariable(mesh=mesh, name="$u_x$")
yVelocity = fp.CellVariable(mesh=mesh, name="$u_y$")

velocity = fp.FaceVariable(mesh=mesh, name=r"$\vec{u}$", rank=1)

xVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.,0.]) + density * gravity[0]
yVelocityEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.,1.]) + density * gravity[1]

ap = fp.CellVariable(mesh=mesh, value=1.)
coeff = 1./ ap.arithmeticFaceValue*mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = fp.DiffusionTerm(coeff=coeff) - velocity.divergence

contrvolume = volumes.arithmeticFaceValue

x, y = mesh.cellCenters
X, Y = mesh.faceCenters

def inlet_velocity(yy):
    return -0.001 * (yy - 3)**2 + 0.009

xVelocity.constrain(inlet_velocity(Y), inlet)
xVelocity.constrain(0., walls)

yVelocity.constrain(0., walls | inlet)

pressureCorrection.constrain(0., top_right)
# pressureCorrection.constrain(0., outlet)

with open(data['residuals.txt'].make().abspath, 'a') as f:
    f.write("{}\t{}\t{}\t{}\t{}\n".format("sweep", "x_residual", "y_residual", "p_residual", "continuity"))

fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                    filename=data["sweep={}.tar.gz".format(0)].make().abspath)

for sweep in range(1, params["sweeps"]+1):
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
    velocity[0, inlet.value] = inlet_velocity(Y)[inlet.value]
    velocity[0, outlet.value] = xVelocity.faceValue[outlet.value]
    velocity[1, outlet.value] = yVelocity.faceValue[outlet.value]

    ## solve the pressure correction equation
    pressureCorrectionEq.cacheRHSvector()
    ## right top point must remain at pressure 0, so no correction
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    ## update the pressure using the corrected value
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection)
    ## update the velocity using the corrected pressure
    xVelocity.setValue(xVelocity - pressureCorrection.grad[0] / \
                                               ap * mesh.cellVolumes)
    yVelocity.setValue(yVelocity - pressureCorrection.grad[1] / \
                                               ap * mesh.cellVolumes)

    if sweep % params["check"] == 0:
        fp.tools.dump.write((xVelocity, yVelocity, velocity, pressure), 
                            filename=data["sweep={}.tar.gz".format(sweep)].make().abspath)
                   
    with open(data['residuals.txt'].make().abspath, 'a') as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format(sweep, xres, yres, pres, max(abs(rhs))))
