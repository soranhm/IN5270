from dolfin import *
import numpy as np
d	rho = 1.0
	u_I = Constant('0')

	def alpha(u):
		return (1+u*u)

	k = []
	h = []
	for i in 10,20,30,50,100:
		u_, t, V,dt = picard(i,[i], alpha, rho,u_I,f= True)
		u_exact = Expression('t*pow(x[0],2)*(0.5-x[0]/3.)',t=t)
		h.append(dt)
		u_e = project(u_exact,V)
		e = (u_e.vector().array() - u_.vector().array())
		E = numpy.sqrt(numpy.sum(e**2)/u_.vector().array().size)
		k.append(E/dt)
	for i in range(len(k)):
		print "h = %3.5f, E/h = %3.7e" %( h[i], k[i] )
def nld_solver(N,I,alpha,dt,T,rho,f,P,user_action=None,u_e=None):
    """
    Solves the non-linear diffusion problem, with Neuman bcs.:
    rho * u_t = div(a(u), grad(u)) + f

    Args:
      N    (list) : With mesh points as elements, i.e. for 2D: [50,50]
      I    (array): Initial condition at t=0
      alpha (func): The diffusion coefficient, may depend on u, i.e. non-linear
      dt   (float): The temporal resolution, aka the time step
      T    (float): Ending time of computation
      rho  (float): Constant in the equation related to the rate of diffusion
      f    (func) : The source term
      P    (int)  : Degree of finite elements

    Returns:
      Solution at final time step, T
    """
    # Only show output that could lead to, or be a problem:
    set_log_level(WARNING)

    # Define Neumann boundary conditions du/dn=0
    # Used by standard

    # Create mesh (1D,2D,3D)
    dim  = np.array(N).size
    mesh = UnitIntervalMesh(N[0]) if dim == 1 else UnitSquareMesh(N[0],N[1]) if dim == 2 \
           else UnitCubeMesh(N[0],N[1],N[2])

    # Create function space V, using 'continous Galerkin'..
    # ..implying the standard Lagrange family of elements
    # P1 = 1, P2 = 2 etc.
    V = FunctionSpace(mesh, 'CG', P)

    # Use initial condition to specify previous (first) known u
    u_1 = interpolate(I, V)

    # Define variables for the equation on variational form
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (u*v + dt/rho*inner(alpha(u_1)*nabla_grad(u), nabla_grad(v)))*dx
    L = (u_1 + dt/rho*f)*v*dx

    # Run simulation:
    Err = [] # Empty list for possible errors
    t = 0; t += dt
    u = Function(V) # The new solution
    while t <= T:
        f.t = t # Update time to current in source term
        # Do a single Picard iteration
        solve(a == L, u)
        u_1.assign(u) # Update u_1 for next iteration
        t += dt
        # Find difference between analytical and numeric sol.
        if u_e:
            u_e.t = t
            u_e_interp = interpolate(u_e, V)
        else:
            u_e_interp = None
        if user_action:
            E = user_action(u, u_e_interp, t-dt, dt)
            Err.append(E)
    if user_action:
        return u_1,V,Err[-1]
    else:
        return u_1,V


def task_d():
    """
    Testing a constant solution
    """
    I = Expression('1.0')# some initial condition
    alpha = lambda u: 2
    f = Constant('0')
    dt = 0.1; T=1; rho = 1
    interval,square,box = [5], [5,5], [5,5,5]
    for P in [1,2]:
        for N in interval,square,box:
            u,V = nld_solver(N,I,alpha,dt,T,rho,f,P)
            u_e = interpolate(I, V)
            abs_err = np.abs(u_e.vector().array() - u.vector().array()).max()
            print 'Using P%d in %dD, err.: %e' % (P,np.array(N).size, abs_err)

def task_e():
    """
    Testing a simple analytical solution
    """
    alpha = lambda u: 1
    f = Constant('0')
    P = 1 # P1 elements
    rho = 1
    dt = 0.1; T = 0.5
    I   = Expression('cos(pi*x[0])')
    u_a = Expression('exp(-pi*pi*t)*cos(pi*x[0])',t=0)
    u_a.t = T
    for counter in range(7):
        h = dt
        N = [int(round(1./np.sqrt(dt)))] * 2 # Times 2 to get "2D list"
        u,V = nld_solver(N,I,alpha,dt,T,rho,f,P)
        u_e = interpolate(u_a, V)

        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2) / u.vector().array().size)
        #print '%.6f %.6f %.6f' %(E,h,E/h)
        print 'h=%.4f, E/h=%.4f, N=%d' %(h, float(E)/h, N[0])
        dt   /= 2 # Divide dt by 2 for next iteration

def task_f():
    """
    Testing a 1D analytical solution with source term adapted (MMS)
    """
    import matplotlib.pyplot as plt
    rho = 1
    I = Constant('0')
    u_exact = Expression('t*pow(x[0],2)*(0.5 - x[0]/3.)',t=0)
    alpha = lambda u: 1 + u**2
    f = Expression('-rho*pow(x[0],3)/3 + rho*pow(x[0],2)/2 + 8*pow(t,3)*pow(x[0],7)/9 - \
                    28*pow(t,3)*pow(x[0],6)/9 + 7*pow(t,3)*pow(x[0],5)/2 - \
                    5*pow(t,3)*pow(x[0],4)/4 + 2*t*x[0] - t', rho=rho, t=0)
    N = [20]
    P = 1 # Use P1 elements
    dt = 0.5;
    for T in [0.5, 1, 1.5]:
        u,V = nld_solver(N,I,alpha,dt,T,rho,f,P)
        u_exact.t = T
        u_e = interpolate(u_exact, V)

        x = np.linspace(0, 1, N[0]+1)
        plt.plot(x, u_e.vector().array()[::-1],'o')
        plt.plot(x, u.vector().array()[::-1])
        plt.legend(['Exact solution','Numerical solution'])
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('t=%s' %T)
        plt.savefig('ft_'+str(int(T*100))+'.png')
        plt.show()

def task_h():
    """
    Verification of solver, by testing convergance rates
    """

    u_exact = Expression("t*x[0]*x[0]*(0.5 - x[0]/3.)", t=0)
    I = Constant("0")
    dt = 0.5; T = 1; rho = 1
    f = Expression("""rho*x[0]*x[0]*(-2*x[0] + 3)/6
                     -(-12*t*x[0] + 3*t*(-2*x[0] + 3))
                     *(pow(x[0], 4)*(-dt + t)*(-dt + t)
                     *(-2*x[0] + 3)*(-2*x[0] + 3) + 36)/324
                     -(-6*t*x[0]*x[0] + 6*t*x[0]
                     *(-2*x[0] + 3))*(36*pow(x[0], 4)
                     *(-dt + t)*(-dt + t)*(2*x[0] - 3)
                     +36*x[0]*x[0]*x[0]*(-dt + t)
                     *(-dt + t)*(-2*x[0] + 3)
                     *(-2*x[0] + 3))/5832""",
                     t=0, dt=dt, rho=rho)
    alpha = lambda u: 1 + u**2
    P = 1 # P1-elements
    N = []

    # Function to compute the error
    def return_error(u, u_e, t, dt):
        e = u_e.vector().array() - u.vector().array()
        E = np.sqrt(np.sum(e**2)/u.vector().array().size).max()
        return E

    list_err = []; dt_values = []

    for i in range(0, 10):
        N = [int(round(1./sqrt(dt)))]
        u,V,E = nld_solver(N,I,alpha,dt,T,rho,f,P,user_action=return_error,u_e=u_exact)
        dt_values.append(dt)
        list_err.append(E)
        dt /= 2.
        f.dt = dt

    #Calculate convergance rates:
    def compute_rates(dt_values, errors):
        m = len(errors)
        r = [np.log(errors[i-1]/errors[i])/
             np.log(dt_values[i-1]/dt_values[i])
             for i in range(1, len(errors))]

        return r

    conv_rates = compute_rates(dt_values, list_err)

    print "\nConvergence rates:"
    for i in range(len(conv_rates)):
        print "h1=%f, h2=%f, r=%f" %(dt_values[i], dt_values[i+1], conv_rates[i])


def task_i():
    """
    Simulate nonlinear diffusion of Gaussian function
    """

    sigma = .5
    I = Expression("exp(-1./(2*sigma*sigma)\
                        *(x[0]*x[0] + x[1]*x[1]))", sigma=sigma)
    T = 0.2; dt = 0.002; f = Constant("0")
    rho = 1.; beta = 10.
    alpha = lambda u: 1 + beta*u**2
    N = [40, 40]
    P = 1

    #Animate the diffusion of the surface:
    def animate_solution(u, u_e, t, dt):
        from time import sleep
        sleep(0.05)
        fig = plot(u)#.set_min_max(0,0.83)
        fig.set_min_max(0,0.83)
        #Save initial state and equilibrium state:
        if t<2*dt or t>T-dt:
            fig.write_png("taski_%s" %t)

    u,V,E = nld_solver(N,I,alpha,dt,T,rho,f,P,user_action=animate_solution)



if __name__ == '__main__':
    """
    Uncomment the task to be done
    """
    #task_d()
    #task_e()
    #task_f()
    #task_h()
    task_i()
