import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


###############################################################################
## Define odeint for a complex number
def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z
###############################################################################


###############################################################################
## Define pulsed Rabi frequencies
sigma_Omega=0.4;
tf =15;
t=np.linspace(0, tf, 1000)
t_delay=-0.73*sigma_Omega; # Time delay between two pulse
Amp_rabi=10; # Amplitude of Rabi frequency
def dydt(y0, t):
    # Assume that the laser is on resonance
    delta=0;
    Delta = 0;
    Omega_S =Amp_rabi*5*np.pi*np.exp(-0.5*((t-tf/2-t_delay)/sigma_Omega)**2)/(sigma_Omega*np.sqrt(2*np.pi));
    Omega_P =Amp_rabi*5*np.pi*np.exp(-0.5*((t-tf/2+t_delay)/sigma_Omega)**2)/(sigma_Omega*np.sqrt(2*np.pi));
    A=-1j*np.array([[0,Omega_P/2,0],
                [Omega_P/2,Delta,Omega_S/2],
                [0,Omega_S/2,delta]]);
    return np.dot(A, y0)

Omega_S_t =Amp_rabi*5*np.pi*np.exp(-0.5*((t-tf/2-t_delay)/sigma_Omega)**2)/(sigma_Omega*np.sqrt(2*np.pi));
Omega_P_t =Amp_rabi*5*np.pi*np.exp(-0.5*((t-tf/2+t_delay)/sigma_Omega)**2)/(sigma_Omega*np.sqrt(2*np.pi));

# Assume that all of population accumulated in state |1>
y0 = np.array([1,0,0])

population= odeintz(dydt, y0, t, rtol=1e-10,atol=1e-8)
y1 = population[:,0]
y2 = population[:,1]
y3 = population[:,2]
y1 = np.real(np.conjugate(y1)*y1) # Population in state |1>
y2 = np.real(np.conjugate(y2)*y2) # Population in state |2>
y3 = np.real(np.conjugate(y3)*y3) # Population in state |3>


###############################################################################
## Plot pulsed Rabi frequencies
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t,Omega_S_t*(sigma_Omega*np.sqrt(2*np.pi))/(5*np.pi),'--', label='S');
ax1.plot(t,Omega_P_t*(sigma_Omega*np.sqrt(2*np.pi))/(5*np.pi), label='P');
ax1.legend(loc=2, prop={'size': 15})
ax1.set(xlabel='Time', ylabel='Pulse')

## Plot populations
ax2.plot(t,y1,'g--', label='1')
ax2.plot(t,y2,'r', label='2')
ax2.plot(t,y3,'b', label='3')
ax2.legend(loc=2, prop={'size': 15})
ax2.set(xlabel='Time', ylabel='Population')
