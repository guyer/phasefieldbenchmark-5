from importlib import import_module
import os
import sys
import yaml

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
gravity = [params["gx"], params["gy"]]
pressureRelaxation = 0.8
velocityRelaxation = 0.5

meshmodule = import_module("mesh{}".format(params["problem"]))

(mesh,
 inlet,
 outlet,
 walls,
 top_right) = meshmodule.mesh_and_boundaries(params)

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

def inlet_velocity(yy):
    return -0.001 * (yy - 3)**2 + 0.009

X, Y = mesh.faceCenters

xVelocity.constrain(inlet_velocity(Y), inlet)
xVelocity.constrain(0., walls)
xVelocity.faceGrad.constrain([[0.], [0.]], outlet)

yVelocity.constrain(0., walls | inlet)

# pressure.constrain(0., top_right)
pressure.faceGrad.constrain([[density * gravity[0]], [0.]], outlet)
pressure.grad.constrain([[density * gravity[0]], [0.]], outlet)
pressureCorrection.constrain(0., top_right)
pressureCorrection.faceGrad.constrain([[0.], [0.]], outlet)
pressureCorrection.grad.constrain([[0.], [0.]], outlet)

# pressureCorrection.constrain(0., outlet)

with open(data['residuals.txt'].make().abspath, 'a') as f:
    f.write("\t".join(["sweep", "x_residual", "y_residual", "p_residual", "continuity"]) + "\n")

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
