""" Problem u_(tt) = (qu_(x))_x + f
Only vectorized versin !!
a) q = 1 + (x -L/2)**4, u(x,t) = cos(xpi/L)xos(wt), w = 1
b) q = 1 + cos((pi*x)/L) , u(x,t) = cos(xpi/L)xos(wt), w = 1"""
from numpy import *

def find_F(L_values,oppg):
    '''If q = 'a' --> using q from 13 a)
       If q = 'b' --> using q from 13 a)    '''
    w = 1 # given
    from sympy import symbols,cos,pi,diff,simplify,lambdify
    x,t,L = symbols('x t L')
    q_sym = 0
    if oppg == 'a':
        q = lambda x: 1 + (x - L/2.)**4 # q from a
    if oppg == 'b':
        q = lambda x: 1 + cos((pi*x)/L)# q from b
    u = lambda x,t:cos((pi*x)/L)*cos(w*t)   # given u

    #using python to diff
    u_tt = diff(u(x,t),t,t)  #u_tt
    u_x  = diff(u(x,t),x)    #u_x
    qu_x_x = diff(q(x)*u_x,x)#(qu_x)_x

    # f = u_tt - (qu_x)_x
    f_sym = simplify(u_tt - qu_x_x).subs(L,L_values)
    print 'f = ',f_sym
    #returning non-symbol values
    return lambdify((x,t),f_sym,modules = 'numpy')

def convergence_rate(u, x, t, n):
    # This is conv. rates
    L   = 1.0 # w = 1
    u_e = cos(pi*x/L)*cos(t[n])
    e = u_e - u
    E = sqrt(dt*sum(e**2))
    E_t_list.append(E)

def I(x):
    # u(x,0) = cos(pi*x/L)cos(w*0)
    return cos(pi*x/L)

def solver(I, V, f, c, U_0, U_L, L, dt, C, T,
           user_action=None, version='vectorized',
           stability_safety_factor=1.0,neuman=False):
    """Solve u_tt=(c^2*u_x)_x + f on (0,L)x(0,T]."""

    Nt = int(round(T/dt))
    t = linspace(0, Nt*dt, Nt+1)      # Mesh points in time

    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in linspace(0, L, 101)])
    dx = dt*c_max/(stability_safety_factor*C)
    Nx = int(round(L/dx))
    x = linspace(0, L, Nx+1)   # Mesh points in space

    if isinstance(c, (float,int)):
        c = zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    # Wrap user-given f, I, V, U_0, U_L if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: zeros(x.shape)
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0

    import time;  t0 = time.clock()  # CPU time measurement

    u   = zeros(Nx+1)
    u_1 = zeros(Nx+1)
    u_2 = zeros(Nx+1)

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

        # Load initial condition into u_1
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])

    if user_action is not None:
        user_action(u_1, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    # Solving for the given problems, with the same intial values
    if neuman == True: # case (57)
        i = Ix[0]
        if U_0 is None:
            # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
            # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
            ip1 = i+1
            im1 = ip1  # i-1 -> i+1
            u[i] = u_1[i] + dt*V(x[i]) - \
                   0.5*C2*(0.5*(q[i] + q[im1])*(u_1[im1] - u_1[i])) + \
                   0.5*dt2*f(x[i], t[0])
        else:
            u[i] = U_0(dt)

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1  # i+1 -> i-1
            u[i] = u_1[i] + dt*V(x[i]) - \
                   0.5*C2*((q[i] + q[im1])*(u_1[im1] - u_1[i])) + \
                   0.5*dt2*f(x[i], t[0])
        else:
            u[i] = U_L(dt)

    elif neuman == False: # case (54)
        i = Ix[0]
        if U_0 is None:
            # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
            # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
            ip1 = i+1
            im1 = ip1  # i-1 -> i+1
            u[i] = u_1[i] + dt*V(x[i]) + 0.5*dt2*f(x[i], t[0]) + \
                   C2*q[i]*(u_1[ip1] - u_1[i])
        else:
            u[i] = U_0(dt)

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1  # i+1 -> i-1
            u[i] = u_1[i] + dt*u_2[i] + 0.5*dt2*f(x[i], t[0]) + \
                   C2*2*q[i]*(u_1[im1] - u_1[i])
        else:
            u[i] = U_L(dt)

        # c) when we have 2 new problems in U_0 and U_L
    elif neuman == None:
        #Neumann == None -->  one-sided difference
        """use_std_neuman_bcs == None --> Third option, one-sided difference approach"""
        i = Ix[0]
        if U_L is None:
            ip1 = i+1
            im1 = ip1  # i-1 -> i+1
            u[i] = u_1[i] + dt*V(x[i]) - \
                   0.5*C2*(0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
                   0.5*dt2*f(x[i], t[0])

        else:
            u[i] = U_0(dt)

        i = Ix[-1]
        if U_0 is None:
            im1 = i-1
            ip1 = im1  # i+1 -> i-1
            u[i] = u_1[i] + dt*V(x[i]) + \
                   0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])) + \
                   0.5*dt2*f(x[i], t[0])
        else:
            u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1)

    # Update data structures for next step.
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        if version == 'scalar':
            for i in Ix[1:-1]:
                u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - \
                        0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
                dt2*f(x[i], t[n])

        if version == 'vectorized':
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
            C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) - \
                0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + \
            dt2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        if neuman == True:
            i = Ix[0]
            if U_0 is None:
                # Set boundary values
                # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
                # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
                ip1 = i+1
                im1 = ip1
                u[i] = - u_2[i] + 2*u_1[i] - \
                       C2*((q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
                       dt2*f(x[i], t[n])
            else:
                u[i] = U_0(t[n+1])

            i = Ix[-1]
            if U_L is None:
                im1 = i-1
                ip1 = im1
                u[i] = - u_2[i] + 2*u_1[i] - \
                       C2*((q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
                       dt2*f(x[i], t[n])

            else:
                u[i] = U_L(t[n+1])

        elif neuman == False: # case 54
            i = Ix[0]
            if U_0 is None:
                # Set boundary values
                # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
                # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
                ip1 = i+1
                im1 = ip1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) + \
                      C2*2*q[i]*(u_1[ip1]-u_1[i])
            else:
                u[i] = U_0(t[n+1])

            i = Ix[-1]
            if U_L is None:
                im1 = i-1
                ip1 = im1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) + \
                       C2*2*q[i]*(u_1[im1] - u_1[i])
            else:
                u[i] = U_L(t[n+1])

        elif neuman == None:
            """neuman == None : Third option, one-sided difference"""
            i = Ix[0]
            if U_L is None:
                # Set boundary values
                ip1 = i+1
                im1 = ip1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) + \
                       C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i]))
            else:
                u[i] = U_0(t[n+1])

            i = Ix[-1]
            if U_0 is None:
                im1 = i-1
                ip1 = im1
                u[i] = - u_2[i] + 2*u_1[i] + dt2*f(x[i], t[n]) - \
                       C2*(0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1]))
            else:
                u[i] = U_L(t[n+1])


        if user_action is not None:
            if user_action(u, x, t, n+1):
                 break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2


    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = t0 - time.clock()
    return cpu_time

case1 = raw_input('Exercise your a,b or c: ')
if case1 == 'c':
    q_s = raw_input('Choose q (a or b): ')

Nx1 = raw_input('Enter Nx (int): ')
Nx1_t = isinstance(Nx1, int)

L = 1; C = 1; q = 0; case = 0;
# False for case 54, true for case 57, None for one sided, 'fourth_technique' for d)
if case1 == 'a':
    case = False
    q_s = 'a'
    prnt = 'case (54)'
elif case1 == 'b':
    case = True
    q_s = 'b'
    prnt = 'case (57)'
elif case1 == 'c':
    case = None
    prnt = 'one sided'
else:
    print case1,' or ', Nx1 , ' is not a valid'


# Taking sqrt of q because the solver uses c**2 (=q)
if q_s == 'a':
    c = lambda x: sqrt(1 + (x - L/2.)**4) # a)
elif q_s == 'b':
    c = lambda x: sqrt(1 + cos((pi*x)/L)) # a)
else:
    print q_s, ' is not a valid q'

f = find_F(L,q_s) # a for q from a), and b for q from b)
def q_sym(c,L):
    from sympy import symbols,pi,cos
    x,t,L = symbols('x t L')
    if c == 'a':
        return 1 + (x - L/2.)**4
    elif c == 'b':
        return 1 + cos((pi*x)/L)
    else:
        return None

Nx1 = int(Nx1) # pick Nx
Nx  = range(50,50+Nx1,50)
E_val = []; h = []

for n in Nx:
    E_t_list = []
    dx = float(L)/n
    dt = C*dx/c(0)
    solver(I=I, V=None, f=f, c=c, U_0=None, U_L=None,
           L=L, dt=dt, C=C, T=3, user_action=convergence_rate, version='vectorized',
           stability_safety_factor=1.0, neuman=case)
    h.append(dt)
    E_val.append(max(E_t_list)) # using the max value

m = len(Nx)
r = []
r_t = []
for i in range(1, m, 1):
    conv = log(E_val[i-1]/E_val[i])/log(h[i-1]/h[i])
    if case1 == 'c':
        r_t.append(abs(conv - 1)) # finding how close to 2 it comes
    else:
        r_t.append(abs(conv - 2))
    r.append(conv)

r_min  = min(r_t)
if case1 == 'c':
    print (' |r - 1| :  %.6f' %r_min)
else:
    print (' |r - 2| :  %.6f' %r_min)

# Print out convergence rates
print 'Nx(i) |  h(i)   |   r(i)  |   Problem   |   q  | '
print '-----------------------------------------------------------------'
for i in range(m-1):
    print "%-3i    %-9.3E    %-5.4f   %s    q = %s" \
        %(Nx[i], h[i], r[i], prnt,q_sym(q_s,L))
