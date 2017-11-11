from textwrap import dedent

import fipy as fp

def mesh_and_boundaries(params):
    """Generate a Gmsh mesh appropriate for the parameters

    """
    mesh = fp.Gmsh2D(dedent('''
        Lx = %(Lx)g;
        Ly = %(Ly)g;
        cellSize = %(cellSize)g;
                         
        Point(1) = {0, 0, 0, cellSize};
        Point(2) = {Lx, 0, 0, cellSize};
        Point(3) = {Lx, Ly, 0, cellSize};
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
    ''' % params))

    X, Y = mesh.faceCenters

    inlet = mesh.physicalFaces["left"]
    outlet = mesh.physicalFaces["right"]
    walls = mesh.physicalFaces["top"] | mesh.physicalFaces["bottom"] | mesh.physicalFaces["hole"]
    top_right = outlet & (Y > max(Y) - params["cellSize"])

    return mesh, inlet, outlet, walls, top_right
