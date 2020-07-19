from fenics import*
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
set_log_level(WARNING)
def Nonlinear_solver(N, I, alpha, rho, f, p = 1,T = 1):
    '''
    Solving : rho*u_t = div(alpha(u),grad(u)) + f
    with u(x,0) = I(x) and du/dt = 0 (Neumann)
    N       :  Mesh points, looks like [8,8]
    I       :  Intial condition for u(x,0)

    alpha   :  Function of u
    rho     :  Constant
    f       :  Function
    dt = 0.1:  Times step
    p = 1   :  P1 element, degree of finit element
    T = 1   : Ending time
    '''

    d = np.array(N).size # finding dimensjon
    # making the mesh
    print('N = ',N)
    print('Dimension = ',d)
    if d == 1:
        mesh = UnitIntervalMesh(N[0])
    elif d ==2:
        mesh = UnitSquareMesh(N[0],N[1])
    else:
        mesh = UnitCubeMesh(N[0],N[1],N[2])

    # V using continous Galerkin and Lagrange family of p elements
    V = FunctionSpace(mesh, 'Lagrange' , p)
    v = TestFunction(V)

    u = TrialFunction(V)
    u_ = Function(V)
    u_1 = interpolate(I,V)
    t = 0
    dt = (1./N[0])*(1./N[0])
    a = (u*v + (dt/rho)*inner(alpha(u_1)*nabla_grad(u),nabla_grad(v))) *dx
    L = (u_1 + (dt/rho)*f)*v *dx

    while t <= T:
        t += dt
        f.t = t # update time in the function
        solve(a == L, u_) # solving the problem
        u_1.assign(u_) # update for next iteration

    return u_, V, t, dt

def oppg_d():
    '''
    A Constant solution in 1D, 2D and 3D
    '''
    alpha = lambda u: 1
    rho = 1
    N = 8
    f = Constant('0')
    I = Expression('1',degree=3)
    intarval = [8]; square = [8,8]; box = [8,8,8]
    p1_elem = [intarval ,square, box]
    p1_type = ['Interval 1D','Square 2D','Box 3D']
    j = 0;
    k = 0
    for N in p1_elem:
        u, V, t, dt = Nonlinear_solver(N, I, alpha, rho, f)
        u_exact = Expression('1',degree = 3)
        u_e = interpolate(u_exact, V)
        diff = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
        tol = 1.0E-5

        msg = 'Solving %s: and get a error: %e \n' % (p1_type[j], diff)
        print(msg)
        j += 1
        if diff < tol:
            k += 1
    if k == 3:
        print('\033[32m ' + 'All types runs without any problem \n' + '\033[0m')

def oppg_e():
    '''
    Simple analytical solution and calculating E/h
    '''
    # Given values
    alpha = lambda u: 1
    rho = 1
    f = Constant('0')
    I = Expression('cos(pi*x[0])',degree = 2)
    N_list = [[10,10],[14,14],[20,20],[28,28],[30,30],[35,35]]
    # Empty lists for the calulated values in each case
    h_list = []
    K_list = []
    for N in N_list:
        u, V, t, dt = Nonlinear_solver(N,I,alpha,rho,f)
        h_list.append(dt) # h = dt
        u_exact = Expression('exp(-pi*pi*t)*cos(pi*x[0])', t = t, degree = 2)
        u_e = interpolate(u_exact,V)
        # given error measuring
        e = u_e.vector().get_local() - u.vector().get_local()
        E = np.sqrt(np.sum(e*e)/u.vector().get_local().size)
        K_list.append(E/dt) # E/h
    for i,(h,k) in enumerate(zip(h_list,K_list)): # Print of results
        print('h = %f, E/h = %.8f, for N = %s' %(h,k,N_list[i]))

def oppg_f():
    #def f(): # calculating f
    #    x, t, rho, dt = sp.symbols('x t rho dt')
    #    def u_simple(x,t):
    #        return x**2*(sp.Rational(1,2) - x/3)*t
        # check the boundary contions
        #for x_point in 0, 1:
        #    print('u_x(%s,t): ' %x_point)
        #    print(sp.diff(u_simple(x, t), x).subs(x, x_point).simplify())
        #print('Initial condition:', u_simple(x, 0)) # I = 0
        # MMS: full nonlinear problem
    #    u = u_simple(x, t)
    #    f = rho*sp.diff(u, t) - sp.diff(alpha(u)*sp.diff(u, x), x)
    #    return f.simplify()
    alpha = lambda u: 1 + u**2
    I = Constant('0') # from the sympy run
    rho = 1
    f = Expression('-rho*pow(x[0],3)/3. + rho*pow(x[0],2)/2. + 8*pow(t,3)*pow(x[0],7)/9. \
        - 28*pow(t,3)*pow(x[0],6)/9. + 7*pow(t,3)*pow(x[0],5)/2. - 5*pow(t,3)*pow(x[0],4)/4.\
         + 2*t*x[0] - t', rho = rho, t = 0, degree = 1)
    T_list = [0.5,1,1.5,8,10]
    for T in T_list:
        N = [8]
        u, V, t, dt = Nonlinear_solver(N,I,alpha,rho,f, T = T)
        u_exact = Expression('t*x[0]*x[0]*(0.5 - x[0]/3.)', t = T, degree = 1)
        u_e = interpolate(u_exact,V)
        # plotting diffrens
        plot(u_e,title =('Exact (blue) vs Picard (red) with T = %.1f') % T)
        plot(u)
        plt.show()


if __name__ == '__main__':
    #oppg_d()
    oppg_e()
    #oppg_f()
    print('Pick one exercise')
