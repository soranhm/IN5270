import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# a) exact solution of u'' = 1 , u(0) = u'(1) = 0
def exact_solver():
    x , A, B = sym.symbols('x A B')
    f = 1 # u'' = f =1
    u_x = sym.integrate(f, (x,0,x)) + A
    u = sym.integrate(u_x, (x,0,x)) + B
    r = sym.solve([u.subs(x,0) - 0,          # x=0 condition
             sym.diff(u,x).subs(x, 1) - 0],  # x=1 condition
                        [A, B])              # unknowns
    # Substitute the integration constants in the solution
    u = u.subs(A, r[A]).subs(B, r[B])
    u = sym.simplify(sym.expand(u))  # to get nice print
    u_f = sym.lambdify([x], u, np)   # make into a function for later
    print "Exact solution: u =", u
    return u, u_f

# b) Galerkin and Least square method to find c
def Galerkin():
    i = sym.symbols('i', integer=True)
    x = sym.symbols('x', real=True)
    psi = sym.sin((2*i+1)*(sym.pi*x/2))
    f = 1
    Aii = ((2*i+1)*(sym.pi/2.))**2 * sym.integrate( psi*psi , (x,0,1))
    b = f*psi
    bi = - sym.integrate( b, (x,0,1))
    ci = bi/Aii
    # for the calculations later
    print 'c in Galerkin: c = ',ci
    c = sym.lambdify([i], ci, np)
    b = sym.lambdify([i,x], psi, np)
    return Aii, c, b

def Least_squares():
    i = sym.symbols('i', integer=True)
    x = sym.symbols('x', real=True)
    psi = sym.sin((2*i+1)*(sym.pi*x/2))
    f = 1 #
    #For i=j:
    Aii = ((2*i+1)*(sym.pi/2.))**4 * sym.integrate( psi*psi , (x,0,1))
    bi = -((2*i+1)*(sym.pi/2))**2 * sym.integrate( f * psi, (x,0,1))
    ci = bi/Aii
    print 'c in Least square: c = ',ci
    # for the calculations later
    c = sym.lambdify([i], ci, np)
    b = sym.lambdify([i,x], psi, np)
    return Aii, c, b

def coeff_dec(method):
    Aii, c, b = method()
    for i in range(0,11): # c is function of i
        c1 = c(i)
        c0 = c(i-1)
        c1_c0 = c1/c0
        print("i=%2i : %.4f" % (i,c1_c0))

# c) Visualize the solutions in b) for N = 0,1,20 for Galerkin
def visualize1():
    Ns = [0,1,20]
    Aii, c, b = Galerkin()
    x = np.linspace(0,1,101)
    # fetching the exact adn numerical solution:
    u, u_e = exact_solver()
    plt.plot(x,u_e(x),label="u_e=%s" % u)
    for N in Ns:
        uj = 0
        for j in range(0,N+1):
            uj += (c(j)*b(j,x))
        plt.plot(x, uj, label="u for N = %.f" % N)
        plt.xlabel('x')
        plt.hold('on')
    plt.legend()
    plt.show()

# d) & e) New basis
def visualize2():
    i = sym.symbols('i', integer=True)
    x = sym.symbols('x', real=True)
    psi = sym.sin((2*i+1)*(sym.pi*x/2))
    f = 1
    Aii = ((i+1)*(sym.pi/2.))**2 * sym.integrate( psi*psi , (x,0,2))
    b = f*psi
    bi = - sym.integrate((b), (x,0,2))
    ci = bi/Aii
    c = sym.lambdify([i], ci, np)
    b = sym.lambdify([i,x], b, np)

    Ns = [0,1,20]
    x = np.linspace(0,2,101)
    u, u_e = exact_solver()
    plt.plot(x,u_e(x), label="u_e=%s" % u)

    for N in Ns:
        uj = 0
        for j in range(0,N+1):
            uj += (c(j)*b(j,x))
        plt.plot(x, uj, label="u for N = %.f" % N)
        plt.xlabel('x')
        plt.hold('on')
    plt.legend()
    plt.show()
    return Aii, c, b

if __name__=="__main__":
    exact_solver()
    Least_squares()
    #Galerkin()
    coeff_dec(Least_squares) # testing for Least_squares
    visualize1()
    #visualize2()
