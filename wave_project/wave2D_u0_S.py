
import time, sys
from numpy import *
from numpy.linalg import norm

def solver(b, I, q, V, f, c, Lx, Ly, Nx, Ny, dt, T,
           user_action=None, version='scalar'):

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Help variables
    dt2 = dt**2

    xv = x[:,newaxis]          # for vectorized function evaluations, same as linspace
    yv = y[newaxis,:]

    stability_limit = (1/float(c))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit

    Nt = int(round(T/float(dt)))
    t = linspace(0, Nt*dt, Nt+1)    # mesh points in time
    Cx2 = (c*dt/dx)**2;  Cy2 = (c*dt/dy)**2    # help variables
    dt2 = dt**2

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((x.shape[0], y.shape[1]))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((x.shape[0], y.shape[1]))


    order = 'C'
    u   = zeros((Nx+1,Ny+1), order=order)   # solution array
    u_1 = zeros((Nx+1,Ny+1), order=order)   # solution at t-dt
    u_2 = zeros((Nx+1,Ny+1), order=order)   # solution at t-2*dt
    f_a = zeros((Nx+1,Ny+1), order=order)   # for vectorized
    V_a = zeros((Nx+1,Ny+1), order=order)
    q_a = zeros((Nx+1, Ny+1), order=order)

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])

    # Load initial condition into u_1
    # Time n = 0
    if version == 'scalar':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
    else: # use vectorized version
        u_1[:,:] = I(xv, yv)


    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(b, q, u, u_1, u_2, f, x, y, t, n,
        Cx2, Cy2, dt2, V, step1=True)

    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        V_a[:,:] = V(xv, yv)
        u = advance_vectorized(b, q_a, u, u_1, u_2, f_a,
        Cx2, Cy2, dt2, V_a, step1=True)

    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2
    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance_scalar(b,q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2)
        else:
            f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
            u = advance_vectorized(b, q_a, u, u_1, u_2, f_a, Cx2, Cy2, dt2)
        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to set u = u_1 if u is to be returned!
    # dt might be computed in this function so return the value
    return u,x,y,t


def advance_scalar(b, q, u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2,
                   V=None, step1=False):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = sqrt(dt2)  # save
    B = (1 + (b*dt*0.5))**(-1)

    if step1:
        I = u_1
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                     b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                     Cx2*((0.5*(q(x[i],y[j]) + q(x[i+1],y[j]))*(I[i+1,j]- I[i,j])) - \
                     0.5*(q(x[i-1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[i-1,j])) + \
                     Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[j+1]))*(I[i,j+1] - I[i,j])) - \
                     0.5*(q(x[i],y[j-1]) + q(x[i],y[j]))*(I[i,j] - I[i,j-1])))
        # The 4 corners
        # bottom left
        j = Iy[0];i = Ix[0]
        ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = jp1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

        # bottom right
        i = Ix[-1];j = Iy[0]
        im1 = i-1; ip1 = im1; jp1 = j+1; jm1 = jp1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                 b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                 Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                 0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                 Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                 0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))


        # top right
        j = Iy[-1];i = Ix[-1]
        im1 = i-1; ip1 = im1; jm1 = j-1; jp1 = jm1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
             b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
             Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
             0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
             Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
             0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

        # top left
        j = Iy[-1];i = Ix[0]
        ip1 = i+1;im1 = ip1;jm1 = j-1;jp1 = jm1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
            b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
            Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
            0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
            Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
            0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

        # Boundary condition
        # y = 0
        j = Iy[0]
        for i in Ix[1:-1]:
            ip1 = i+1; im1 = i-1; jp1 = j+1; jm1=jp1
            u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                 b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                 Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                 0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                 Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                 0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

        # x = Nx
        i = Ix[-1]
        for j in Iy[1:-1]:
            im1 = i-1
            ip1 = im1
            jp1 = j+1
            jm1 = j-1
            u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                 b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                 Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                 0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                 Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                 0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

        # y = Ny
        j = Iy[-1]
        for i in Ix[1:-1]:
            ip1 = i+1; im1 = i-1; jm1 = j-1; jp1 = jm1
            u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))
        # x = 0
        i = Ix[0]
        for j in Iy[1:-1]:
            ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = j-1
            u[i,j] = 0.5*(2*I[i,j] + 2*dt*V(x[i], y[j]) - \
                b*dt2*V(x[i], y[j]) + dt2*f(x[i], y[j], 0) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(I[ip1,j]- I[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(I[i,jp1] - I[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(I[i,j] - I[i,jm1])))

    else:
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[i+1],y[j]))*(u_1[i+1,j] - u_1[i,j])) - \
                0.5*(q(x[i-1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i-1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[j+1]))*(u_1[i,j+1] - u_1[i,j])) - \
                0.5*(q(x[i],y[j-1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,j-1])))

        # The 4 corners
        # bootom left
        j = Iy[0]; i = Ix[0]
        ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = jp1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

        # top left
        j = Iy[-1];i = Ix[0]
        ip1 = i+1; im1 = ip1; jm1 = j-1; jp1 = jm1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

        # top right
        j = Iy[-1]; i = Ix[-1]
        im1 = i-1; ip1 = im1; jm1 = j-1; jp1 = jm1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

        # bottom right
        i = Ix[-1]; j = Iy[0]
        im1 = i-1; ip1 = im1; jp1 = j+1; jm1 = jp1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

        # Boundary condition
        # y = 0
        j = Iy[0]
        for i in Ix[1:-1]:
            ip1 = i+1; im1 = i-1; jp1 = j+1; jm1=jp1
            u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

        # x = Nx
        i = Ix[-1]
        for j in Iy[1:-1]:
            im1 = i-1; ip1 = im1; jp1 = j+1; jm1 = j-1
            u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))


        # y = Ny
        j = Iy[-1]
        for i in Ix[1:-1]:
            ip1 = i+1; im1 = i-1; jm1 = j-1; jp1 = jm1
            u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))


        # x = 0
        i = Ix[0]
        for j in Iy[1:-1]:
            ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = j-1
            u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f(x[i], y[j], t[n]) + \
                Cx2*((0.5*(q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q(x[im1],y[j]) + q(x[i],y[j]))*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q(x[i],y[j]) + q(x[i],y[jp1]))*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q(x[i],y[jm1]) + q(x[i],y[j]))*(u_1[i,j] - u_1[i,jm1])))

    return u

def advance_vectorized(b, q_a, u, u_1, u_2, f_a, Cx2, Cy2, dt2,
                       V_a=None, step1=False):
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = sqrt(dt2)
    B = (1 + (b*dt*0.5))**(-1)
    if step1:
        I = u_1[:,:]
        u[1:-1,1:-1] = 0.5*(2*I[1:-1,1:-1] + 2*dt*V_a[1:-1,1:-1] - \
                 b*dt2*V_a[1:-1,1:-1] + dt2*f_a[1:-1,1:-1] + \
                 Cx2*((0.5*(q_a[1:-1,1:-1] + q_a[2:,1:-1])*(I[2:,1:-1]- I[1:-1,1:-1])) - \
                 0.5*(q_a[:-2,1:-1] + q_a[1:-1,1:-1])*(I[1:-1,1:-1] - I[:-2,1:-1])) + \
                 Cy2*((0.5*(q_a[1:-1,1:-1] + q_a[1:-1,2:])*(I[1:-1,2:] - I[1:-1,1:-1])) - \
                 0.5*(q_a[1:-1,:-2] + q_a[1:-1,1:-1])*(I[1:-1,1:-1] - I[1:-1,:-2])))
        # the 4 corners
        # bottom left
        j = Iy[0];i = Ix[0]
        ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = jp1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V_a[i,j] - \
                b*dt2*V_a[i,j] + dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(I[ip1,j]- I[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(I[i,jp1] - I[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(I[i,j] - I[i,jm1])))

        # bottom right
        i = Ix[-1]; j = Iy[0]
        im1 = i-1; ip1 = im1; jp1 = j+1; jm1 = jp1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V_a[i,j] - \
                b*dt2*V_a[i,j] + dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(I[ip1,j]- I[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(I[i,jp1] - I[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(I[i,j] - I[i,jm1])))

        # top right
        j = Iy[-1];i = Ix[-1]
        im1 = i-1; ip1 = im1; jm1 = j-1; jp1 = jm1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V_a[i,j] - \
                b*dt2*V_a[i,j] + dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(I[ip1,j]- I[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(I[i,jp1] - I[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(I[i,j] - I[i,jm1])))

        # top left
        j = Iy[-1];i = Ix[0]
        ip1 = i+1; im1 = ip1; jm1 = j-1; jp1 = jm1
        u[i,j] = 0.5*(2*I[i,j] + 2*dt*V_a[i,j] - \
                b*dt2*V_a[i,j] + dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(I[ip1,j]- I[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(I[i,j] - I[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(I[i,jp1] - I[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(I[i,j] - I[i,jm1])))

        # Boundary conditions
        # y = 0
        j = Iy[0]
        u[1:-1,j] = 0.5*(2*I[1:-1,j] + 2*dt*V_a[1:-1,j] - b*dt2*V_a[1:-1,j] + dt2*f_a[1:-1,j] + \
             Cx2*((0.5* (q_a[1:-1,j] + q_a[2:,j])       *(I[2:,j]     - I[1:-1,j])) - \
             0.5*       (q_a[:-2,j]  + q_a[1:-1,j])     *(I[1:-1,j]   - I[:-2,j])) + \
             Cy2*((0.5* (q_a[1:-1,j] + q_a[1:-1,j+1])   *(I[1:-1,j+1] - I[1:-1,j])) - \
             0.5*(q_a[1:-1,j+1] + q_a[1:-1,j])*(I[1:-1,j] - I[1:-1,j+1])))

        # x = Nx
        i = Ix[-1]
        u[i,1:-1] = 0.5*(2*I[i,1:-1] + 2*dt*V_a[i,1:-1] - b*dt2*V_a[i,1:-1] + dt2*f_a[i,1:-1] + \
             Cx2*((0.5*(q_a[i,1:-1] + q_a[i-1,1:-1])*(I[i-1,1:-1]- I[i,1:-1])) - \
             0.5*(q_a[i-1,1:-1] + q_a[i,1:-1])*(I[i,1:-1] - I[i-1,1:-1])) + \
             Cy2*((0.5*(q_a[i,1:-1] + q_a[i,2:])*(I[i,2:] - I[i,1:-1])) - \
             0.5*(q_a[i,:-2] + q_a[i,1:-1])*(I[i,1:-1] - I[i,:-2])))

        # y = Ny
        j = Iy[-1]
        u[1:-1,j] = 0.5*(2*I[1:-1,j] + 2*dt*V_a[1:-1,j] - b*dt2*V_a[1:-1,j] + dt2*f_a[1:-1,j] + \
             Cx2*((0.5* (q_a[1:-1,j] + q_a[2:,j])       *(I[2:,j]     - I[1:-1,j])) - \
             0.5*       (q_a[:-2,j]  + q_a[1:-1,j])     *(I[1:-1,j]   - I[:-2,j])) + \
             Cy2*((0.5* (q_a[1:-1,j] + q_a[1:-1,j-1])   *(I[1:-1,j-1] - I[1:-1,j])) - \
             0.5*(q_a[1:-1,j-1] + q_a[1:-1,j])*(I[1:-1,j] - I[1:-1,j-1])))

        # x = 0
        i = Ix[0]
        u[i,1:-1] = 0.5*(2*I[i,1:-1] + 2*dt*V_a[i,1:-1] - b*dt2*V_a[i,1:-1] + dt2*f_a[i,1:-1] + \
                 Cx2*((0.5*(q_a[i,1:-1] + q_a[i+1,1:-1])*(I[i+1,1:-1]- I[i,1:-1])) - \
                 0.5*(q_a[i+1,1:-1] + q_a[i,1:-1])*(I[i,1:-1] - I[i+1,1:-1])) + \
                 Cy2*((0.5*(q_a[i,1:-1] + q_a[i,2:])*(I[i,2:] - I[i,1:-1])) - \
                 0.5*(q_a[i,:-2] + q_a[i,1:-1])*(I[i,1:-1] - I[i,:-2])))


    else:
        u[1:-1,1:-1] = B*(2*u_1[1:-1,1:-1] - u_2[1:-1,1:-1] + \
                (b*dt/2)*u_2[1:-1,1:-1] + dt2*f_a[1:-1,1:-1] + \
                Cx2*((0.5*(q_a[1:-1,1:-1] + q_a[2:,1:-1])*(u_1[2:,1:-1] - u_1[1:-1,1:-1])) - \
                0.5*(q_a[:-2,1:-1] + q_a[1:-1,1:-1])*(u_1[1:-1,1:-1] - u_1[:-2,1:-1])) + \
                Cy2*((0.5*(q_a[1:-1,1:-1] + q_a[1:-1,2:])*(u_1[1:-1,2:] - u_1[1:-1,1:-1])) - \
                0.5*(q_a[1:-1,:-2] + q_a[1:-1,1:-1])*(u_1[1:-1,1:-1] - u_1[1:-1,:-2])))

        # the 4 corners
        # bottom left
        j = Iy[0]; i = Ix[0]
        ip1 = i+1; im1 = ip1; jp1 = j+1; jm1 = jp1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(u_1[i,j] - u_1[i,jm1])))

        # bottom right
        i = Ix[-1]; j = Iy[0]
        im1 = i-1; ip1 = im1; jp1 = j+1; jm1 = jp1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(u_1[i,j] - u_1[i,jm1])))

        # top left
        j = Iy[-1]; i = Ix[0]
        ip1 = i+1; im1 = ip1; jm1 = j-1; jp1 = jm1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(u_1[i,j] - u_1[i,jm1])))

        # top right
        j = Iy[-1]; i = Ix[-1]
        im1 = i-1; ip1 = im1; jm1 = j-1; jp1 = jm1
        u[i,j] = B*(2*u_1[i,j] - u_2[i,j] + (b*dt/2)*u_2[i,j] + \
                dt2*f_a[i,j] + \
                Cx2*((0.5*(q_a[i,j] + q_a[ip1,j])*(u_1[ip1,j] - u_1[i,j])) - \
                0.5*(q_a[im1,j] + q_a[i,j])*(u_1[i,j] - u_1[im1,j])) + \
                Cy2*((0.5*(q_a[i,j] + q_a[i,jp1])*(u_1[i,jp1] - u_1[i,j])) - \
                0.5*(q_a[i,jm1] + q_a[i,j])*(u_1[i,j] - u_1[i,jm1])))

        # Boundary conditions
        # y = 0
        j = Iy[0]
        u[1:-1,j] = B*(2*u_1[1:-1,j] - u_2[1:-1,j] + \
                (b*dt/2)*u_2[1:-1,j] + dt2*f_a[1:-1,j] + \
                Cx2*((0.5*(q_a[1:-1,j] + q_a[2:,j])*(u_1[2:,j] - u_1[1:-1,j])) - \
                0.5*(q_a[:-2,j] + q_a[1:-1,j])*(u_1[1:-1,j] - u_1[:-2,j])) + \
                Cy2*((0.5*(q_a[1:-1,j] + q_a[1:-1,j+1])*(u_1[1:-1,j+1] - u_1[1:-1,j])) - \
                0.5*(q_a[1:-1,j+1] + q_a[1:-1,j])*(u_1[1:-1,j] - u_1[1:-1,j+1])))

        # x = Nx
        i = Ix[-1]
        u[i,1:-1] = B*(2*u_1[i,1:-1] - u_2[i,1:-1] + \
                (b*dt/2)*u_2[i,1:-1] + dt2*f_a[i,1:-1] + \
                Cx2*((0.5*(q_a[i,1:-1] + q_a[i-1,1:-1])*(u_1[i-1,1:-1] - u_1[i,1:-1])) - \
                0.5*(q_a[i-1,1:-1] + q_a[i,1:-1])*(u_1[i,1:-1] - u_1[i-1,1:-1])) + \
                Cy2*((0.5*(q_a[i,1:-1] + q_a[i,2:])*(u_1[i,2:] - u_1[i,1:-1])) - \
                0.5*(q_a[i,:-2] + q_a[i,1:-1])*(u_1[i,1:-1] - u_1[i,:-2])))

        # y = Ny
        j = Iy[-1]
        u[1:-1,j] = B*(2*u_1[1:-1,j] - u_2[1:-1,j] + \
                (b*dt/2)*u_2[1:-1,j] + dt2*f_a[1:-1,j] + \
                Cx2*((0.5*(q_a[1:-1,j] + q_a[2:,j])*(u_1[2:,j] - u_1[1:-1,j])) - \
                0.5*(q_a[:-2,j] + q_a[1:-1,j])*(u_1[1:-1,j] - u_1[:-2,j])) + \
                Cy2*((0.5*(q_a[1:-1,j] + q_a[1:-1,j-1])*(u_1[1:-1,j-1] - u_1[1:-1,j])) - \
                0.5*(q_a[1:-1,j-1] + q_a[1:-1,j])*(u_1[1:-1,j] - u_1[1:-1,j-1])))

        # x = 0
        i = Ix[0]
        u[i,1:-1] = B*(2*u_1[i,1:-1] - u_2[i,1:-1] + \
                (b*dt/2)*u_2[i,1:-1] + dt2*f_a[i,1:-1] + \
                Cx2*((0.5*(q_a[i,1:-1] + q_a[i+1,1:-1])*(u_1[i+1,1:-1] - u_1[i,1:-1])) - \
                0.5*(q_a[i+1,1:-1] + q_a[i,1:-1])*(u_1[i,1:-1] - u_1[i+1,1:-1])) + \
                Cy2*((0.5*(q_a[i,1:-1] + q_a[i,2:])*(u_1[i,2:] - u_1[i,1:-1])) - \
                0.5*(q_a[i,:-2] + q_a[i,1:-1])*(u_1[i,1:-1] - u_1[i,:-2])))

    return u



def constant(Nx, Ny, version):
    """Exact discrete solution of the scheme."""
    Lx = 5;  Ly = 2
    c = 3
    dt = 0.01
    T = 18
    b = 0.5

    #functions
    exact_solution = lambda x,y,t: c
    I = lambda x,y : exact_solution(x,y,0)
    q = lambda x,y :c
    V = lambda x,y : 0
    f = lambda x,y,t : 0

    def assert_no_error(u, x, xv, y, yv, t, n):
        u_e  = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        E.append(diff)
        tol  = 1E-13
        msg  = "diff=%g, step=%d, time=%g" %(diff, n, t[n])
        #print('diff=%g, step=%d, time=%g' %(diff, n, t[n]))
        succes = diff < tol
        assert succes, msg

    u,x,y,new_dt = solver(
        b, I, q, V, f, c, Lx, Ly, Nx, Ny, dt, T,
        user_action=assert_no_error, version=version)
    return new_dt



def test_constant():
    # Test a diffrent mesh where Nx > Ny and Nx < Ny
    global E
    versions = 'scalar','vectorized'
    for Nx in range(2, 6, 2):
        for Ny in range(2, 6, 2):
            for ver in versions:
                E = [] # Will contain error from computation
                print 'testing', ver, 'for %dx%d mesh' % (Nx, Ny)
                constant(Nx, Ny, ver)
                print '- largest error:', max(E)

def test_plug():

    #Check that an initial plug, after one step.
    Lx = 1.1; Ly = 1.1
    Nx = 11;  Ny = 11
    dt = 0.1;  T = 1.1
    b  = 0 # No damping
    c = 1
    V = lambda x, y: 0.0
    f = lambda x, y, t: 0.0
    q = lambda x, y: ones((Nx+1,Ny+1))*c # needs to be function
    Lx=1.0; Ly=1.0
    Nx=9; Ny=9; T=1.0

    x = linspace(0,Lx,Nx+1)
    y = linspace(0,Ly,Ny+1)
    Nt = int(round(T/float(dt)))
    t = linspace(0, Nt*dt, Nt+1)

    #the functions
    def I(x, y):
        if y.all <= 0.6 and  y.all >= 0.4:
            I = 1
        else:
            I = 0
        return I


    u_num,x,y,t = solver(b, I, q , V, f, c, Lx, Ly, Nx, Ny, dt, T,version='vectorized')
    u_exact = array([[I(x, y_) for y_ in y] for x_ in x])
    tol = 1E-12
    step_num = t/dt
    diff = abs(u_exact - u_num).max()
    for i in t:
        msg = 'diff=%g at time=%g' % (diff, i)
        succes = diff < tol
        assert succes, msg

class FindError:
    # generall error
    def __init__(self, ue):
        self.ue = ue
        self.E  = 0
        self.h  = 0

    def __call__(self, u, x, xv, y, yv, t, n):
        if n == 0:
            self.h = t[1] - t[0]
        last_timestep = len(t)-1
        if n == last_timestep:
            dx = x[1] - x[0];  dy = y[1] - y[0]
            X,Y    = meshgrid(xv,yv)
            u_e    = self.ue(X,Y,t[n])
            u_e    = u_e.transpose()
            e = norm(u-u_e)
            self.E = sqrt(dx*dy)*e


def test_standing_undamped_waves():
    A=0.2; mx=2.; my=3.; Lx=1.; Ly=1.;
    kx      = mx*pi/Lx; ky = my*pi/Ly
    w       = sqrt(kx**2 + ky**2)
    b = 0 # no damping

    ue = lambda x,y,t: A*cos(kx*x)*cos(ky*y)*cos(w*t)
    I  = lambda x,y  : ue(x,y,0)
    V  = lambda x,y,  : 0
    f  = lambda x,y,t: 0
    c  =  1
    q  = lambda x,y  : c

    E = []; h = []; T = 0.4; dt = -1
    Nx_values = [5,10,20,40,80,160,320]

    for i,Nx in enumerate(Nx_values):
        Ny = Nx
        error = FindError(ue)
        u_num2,x2,y2,t2 = solver(b, I, q, V, f, c, Lx, Ly, Nx, Ny, dt, T,
                   user_action=error, version='vectorized')

        E.append( error.E )
        h.append( error.h )

    # Print out convergence rates
    print "---------------------------------------------------------"
    print "N(i) |   dt(i)    |   r(i)"
    print "---------------------------------------------------------"
    m = len(Nx_values)
    r = [log(E[i-1]/E[i])/log(h[i-1]/h[i]) for i in range(1, m, 1)]
    r = [round(r_, 2) for r_ in r] # Round to 2 decimals
    for i in range(m-1):
        print "%-3i  %-9.3E   %-5.2f" \
            %(Nx_values[i], h[i], r[i])

    tol = 0.05
    assert r[-1]-2 < tol # Check that we converge on r = 2

def test_manufactured_solution(): # NOT DONE (Hard to understand)
    # trying to find q and f, with given u_e
    import sympy as s
    x,y,t,A,B,c,w,kx,ky = s.symbols('x y t A B c w kx ky')
    q = s.exp(-c*t)
    u = (A*s.cos(w*t) + B*s.sin(w*t))*s.exp(-c*t)*s.cos(kx*x)*s.cos(ky*y)
    f = 0
    lhs =  s.diff(u,t,t)
    rhs_x = s.diff(q*s.diff(u,x),x)
    rhs_y = s.diff(q*s.diff(u,y),y)
    r = rhs_x + rhs_y + f - lhs
    print s.simplify(r)

def physical_problem(problem):
    # Variables
    c = 1;T = 0.5; dt = 0.001
    Lx = 1.; Ly = 1. ;Nx = 100; Ny = 100
    g = 9.81    # accelration
    f = lambda x, y, t: 0.
    V = lambda x, y: 0.
    b = 1.      # b = 1 gives a circular contour lines, else an elliptic shape with elliptic contour
    I0 = 1.1    # Depth
    Ia = 0.4
    Im = 0.     # the loc of peak
    Is = 0.1    # width of function
    B0 = 0.     # Ocean floor
    Ba = 0.7    # Height of top
    Bmx = Lx/2. # top at x axis
    Bmy = Ly/2. # top at y axis
    Bs = 0.2    # steepnes

    # Gaussian problem:
    if problem == 'gaussian':
        I = lambda x, y: I0 + Ia*exp(-((x - Im)/Is)**2)
        B = lambda x, y: B0 + Ba*exp(-((x - Bmx)/Bs)**2 - ((y - Bmy)/(b*Bs))**2)

    # Cosine hat:
    elif problem == 'cosine_hat':
        I = lambda x, y: I0 + Ia*exp(-((x - Im)/Is)**2)
        B = lambda x, y: B0 + Ba*cos(pi*(x - Bmx)/(2*Bs))*cos(pi*(y - Bmy)/(2*Bs)) \
            if 0 <= np.sqrt((x - Bmx)**2+(y - Bmy)**2) <= Bs else B0


    # Box:
    elif problem == 'box':
        I = lambda x, y: I0 + Ia*exp(-((x - Im)/Is)**2)
        B = lambda x, y: B0 + Ba if Bmx - Bs <= x <= Bmx + Bs and Bmy-b*Bs <=y<= Bmy+b*Bs\
            else B0

    q = lambda x, y: sqrt(g*(I(x,y)-B(x,y)))

    # getting the values from solver (should added a plot)
    u_num,x,y,t = solver(b, I, q, V, f, c, Lx, Ly, Nx, Ny, dt, T,version='vectorized')


if __name__ == '__main__':
    #test_constant()
    test_plug()
    #test_standing_undamped_waves()
    #test_manufactured_solution()
    #physical_problem('gaussian')
    #physical_problem('cosine_hat')
    #physical_problem('box')
