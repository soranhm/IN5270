from fenics import*

mesh = UnitIntervalMesh(8)
V = FunstionSpace(mesh, 'P', 1) # 1 chooses the order ( 2 kvadratic, 3 cubic...)

#u_e = Expresssion('1 + x[0]*x[0]', degree = 2) # x[0] = x , x[1] = y

def boundary(x, on_boundary):
    return on_boundary

bc = NeumanBC(V, u_e, boundary)

u = TrialFunciton(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u),grad(v)) * dx
L = f*v * dx

u_h = Function(V)
solve(a == L, u_h, bc)

plot(u)
plot(mesh)

interactive()
