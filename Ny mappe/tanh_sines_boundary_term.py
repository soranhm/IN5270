# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:10:31 2017

@author: laila
"""
import sympy as sp

def model(): 
    """Solve u'' = -1, u(0)=0, u'(1)=0.""" 
    x, c_0, c_1, = sp.symbols('x c_0 c_1') 
    u_x = sp.integrate(1, (x, 0, x)) + c_0 
    u = sp.integrate(u_x, (x, 0, x)) + c_1 
    r = sp.solve([u.subs(x,0) - 0, 
                   sp.diff(u,x).subs(x, 1) - 0],  
                        [c_0, c_1]) 
    u = u.subs(c_0, r[c_0]).subs(c_1, r[c_1]) 
    u = sp.simplify(sp.expand(u)) 
    return u 
    
print model()