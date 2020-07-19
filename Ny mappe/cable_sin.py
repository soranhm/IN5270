# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:17:10 2017

@author: laila

Exercise 2: Compute the deflection of a cable with sine functions
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.abc import q



"""a) Solve u'' = 1, u(0)=0, u'(1)=0."""
def symbolic_solver():
    x, c_0, c_1, = sp.symbols('x c_0 c_1')
    u_x = sp.integrate(1, (x, 0, x)) + c_0
    u = sp.integrate(u_x, (x, 0, x)) + c_1
    r = sp.solve([u.subs(x,0) - 0,          # x=0 condition
             sp.diff(u,x).subs(x, 1) - 0],  # x=1 condition
                        [c_0, c_1])         # unknowns
    # Substitute the integration constants in the solution
    u = u.subs(c_0, r[c_0]).subs(c_1, r[c_1])
    u = sp.simplify(sp.expand(u))
    #for the calculations later
    un = sp.lambdify([x], u, np)
    print "The exact solution is u =", u
    return u, un


"""b) Use a Galerkin and least squares method to find the coefficients"""

def least_squares():
    i = sp.symbols('i', integer=True)
    x = sp.symbols('x', real=True)
    #For i=j:
    Aii = ((2*i+1)*(sp.pi/2.))**4 * sp.integrate( (sp.sin((2*i+1)*(sp.pi*x/2)))**2 , (x,0,1))
    bi = -((2*i+1)*(sp.pi/2))**2 * sp.integrate( (sp.sin((2*i + 1)*(sp.pi*x/2))), (x,0,1))
    ci = bi/Aii
    print "Least squares method: ci = ", ci
    return Aii, bi, ci


def galerkin():
    i = sp.symbols('i', integer=True)
    x = sp.symbols('x', real=True)
    Aii = ((2*i+1)*(sp.pi/2.))**2 * sp.integrate( (sp.cos((2*i+1)*(sp.pi*x/2)))**2 , (x,0,1))
    b = sp.sin((2*i+1)*(sp.pi*x/2) )
    bi = - sp.integrate((b), (x,0,1))
    ci = bi/Aii
    print "The Garlerkin method: ci = ", ci
    #for the calculations later
    c = sp.lambdify([i], ci, np)
    b = sp.lambdify([i,x], b, np)
    return Aii, bi, ci, c, b

def dec_coeff():
    Aii, bi, ci, c, b = galerkin()
    for i in range(0,11):
        c1 = c(i)       # c[i] = - 16/((np.pi*(2*i+1))**3)
        c0 = c(i-1)     # c[i-1] =- 16/((np.pi*(2*i+3))**3)
        print("i=%.f : %f" % (i,c1/c0))

"""c)Visualize the solutions in b) for N=0,1,20. """
def visualize1():
    N_list = [0,1,20]
    Aii, bi, ci, c, b = galerkin()
    x = np.linspace(0,1,100)
    #fetching the exact solution:
    u, u_e = symbolic_solver()
    plt.plot(x,u_e(x),label="u_e=%s" % u)
    for N in N_list:
        uj = 0
        for j in range(0,N+1):
            uj += (c(j)*b(j,x))
        plt.plot(x, uj, label="u for N = %.f" % N)
        plt.xlabel('x')
        plt.hold('on')
    plt.legend()
    plt.show()

""""d) & e) New basis"""
def visualize2():
    i = sp.symbols('i', integer=True)
    x = sp.symbols('x', real=True)
    Aii = ((i+1)*(sp.pi/2.))**2 * sp.integrate( (sp.cos((i+1)*(sp.pi*x/2)))**2 , (x,0,2))
    b = sp.sin((i+1)*(sp.pi*x/2) )
    bi = - sp.integrate((b), (x,0,2))
    ci = bi/Aii
    c = sp.lambdify([i], ci, np)
    b = sp.lambdify([i,x], b, np)

    N_list = [0,1,20]
    x = np.linspace(0,2,100)
    u, u_e = symbolic_solver()
    plt.plot(x,u_e(x), label="u_e=%s" % u)

    for N in N_list:
        uj = 0
        for j in range(0,N+1):
            uj += (c(j)*b(j,x))
            print uj
        plt.plot(x, uj, label="u for N = %.f" % N)
        plt.xlabel('x')
        plt.hold('on')
    plt.legend()
    plt.show()
    return Aii, bi, ci, c, b


if __name__=="__main__":
    symbolic_solver()
    least_squares()
    galerkin()
    dec_coeff()
    #visualize1()
    #visualize2()
