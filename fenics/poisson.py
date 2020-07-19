from fenics import*
import matplotlib.pyplot as plt

#mesh = UnitIntervalMesh(8)
mesh = UnitSquareMesh(8,8)  # two dim
V = FunctionSpace(mesh, 'P', 1) # 1 chooses the order ( 2 kvadratic, 3 cubic...)

#u_e = Expresssion('1 + x[0]*x[0]', degree = 2) # x[0] = x , x[1] = y
u_e = Expression('1 + x[0]*x[0] + x[1]*x[1]', degree = 2) # x[0] = x , x[1] = y

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_e, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u),grad(v)) * dx
L = f*v * dx

u_h = Function(V)
solve(a == L, u_h, bc)

plot(u_h)
plot(mesh)
plt.show()
