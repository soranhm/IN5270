import sympy as sym
import numpy as np

V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
c = V; d = I
f = None # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + (w**2)*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = u(t + dt) - I + 0.5*dt**2*w**2*I - dt*V - 0.5*dt**2*f.subs(t,0)
    R = R.subs(t, 0)  # t=0 in the rhs of the first step eq.
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt) - 2*u(t) + u(t-dt))/(dt**2)

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '=== Testing exact solution: %s ===' % u(t)
    print "Initial conditions u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))
    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))
    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)

def linear():
    main(lambda t: V*t + I)

def quadratic():
    b = sym.Symbol('b')  # constant in quadratic
    main(lambda t: b*t**2 + c*t + d)

def polynomial_3degree():
    a,b = sym.symbols('a b')  # constants
    main(lambda t: a*t**3 + b*t**2 + c*t + d)

def solver(I, V, f, w, dt, T):
    """
    Solve u = -w**2*u + f for t in (0,T], u(0)=I and u'(0)=V
    """
    dt = float(dt)
    Nt = int(round(T/dt))
    u = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    u[0] = I
    u[1] = u[0] - 0.5*dt**2*w**2*u[0]  + dt*V + 0.5*dt**2*f(t[0])
    for n in range(1,Nt):
        u[n+1] = dt**2*f(t[n]) + 2*u[n] - u[n-1] - dt**2*w**2*u[n]
    return u,t

def test_quadratic_solution():
    """Verify solver with quadratic, transform from symbol to
    numbers with global and lambdify"""
    global b, w, V, I, T, f, t
    b, V, I, w, T= 2.3, 0.9, 1.2, 1.5, 3 # using some known answers

    u_e = lambda t: b*t**2 + I + V*t
    f = ode_source_term(u_e)
    f = sym.lambdify(t, f) # make t in f into numbers

    dt = 2./w  # number of time steps
    u, t = solver(I=I, w=w, V=V, f=f, dt=dt, T=T)
    u_e = u_e(t)
    diff = np.abs(u - u_e).max()
    tol = 1E-15  # tolerance for comparing floats
    success = diff < tol
    msg = 'Error in quadratic solution:',diff # This is shown when success = False
    assert success, msg

if __name__ == '__main__':
    linear()
    quadratic()
    polynomial_3degree()
    test_quadratic_solution()
