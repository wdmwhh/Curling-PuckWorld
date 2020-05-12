import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import MultipleLocator


def is_illegal(pos, r, env, eps):
    if not (isinstance(pos, (tuple, list))
              and len(pos) == 2):
        raise ValueError("pos should be a tuple or list.")
    
    env_w, env_h = env
    x, y = pos

    check = 0 # 0, 1, 2, 4, 8, 3, 12, 9, 6
    if x < r - eps: check |= 1
    if y < r - eps: check |= 2
    if x > env_w - r + eps: check |= 4
    if y > env_h - r + eps: check |= 8

    return check

def clamp(pos, r, env):
    env_w, env_h = env
    x, y = pos
    if x < r:
        warnings.warn(
            'clamp x from %f to %f' % (x, r))
        x = r
    elif x > env_w - r:
        warnings.warn(
            'clamp x from %f to %f' % (x, env_w - r))
        x = env_w - r
    if y < r:
        warnings.warn(
            'clamp y from %f to %f' % (y, r))
        y = r
    elif y > env_h - r:
        warnings.warn(
            'clamp y from %f to %f' % (y, env_h - r))
        y = env_h - r
    return x, y

def solve_quad_eq(a, b, c):  #at^2 + bt  = c
    if a == 0: return c / b
    if c == 0: return 0
    if a < 0:
        a, b, c = -a, -b, -c
    delta = b * b + 4 * a * c
    if delta < 0:
        raise ValueError(
            "No solutions for the equation.")
    sq_d = np.sqrt(delta)
    t = -b - sq_d if -b - sq_d >= 0 else -b + sq_d
    t /= 2.0 * a
    if t < 0:
        print('at^2 + bt', a * t * t + b * t)
        print('c', c)
        raise ValueError(
            "t = %f < 0. There are some bugs", t)

    return t 

def default_action(pos, target):
    x, y = pos
    tx, ty = target
    a = 0
    if x > tx:
        a |= 1
    if y > ty:
        a |= 2
    return a

# def egreedy_strategy(q_func, s, num_action, 
#     pos=None, target=None, eps=0, default=True):
#     if default and pos and target:
#         raise ValueError
#     max_a = default_action(pos, target)
#     max_q = float(q_func(s, max_a))
#     for a in range(num_action):
#         if a == max_a: continue
#         q = float(q_func(s, a))
#         if q > max_q:
#             max_a = a
#             max_q = q
#     if eps > 0 and np.random.random() < eps:
#         return np.random.randint(num_action), None
#     return max_a, max_q

def egreedy_strategy(q_s, num_action, eps=0):
    max_a = q_s.argmax()
    if eps > 0 and np.random.random() < eps:
        return np.random.randint(num_action)
    return max_a

def plot(episodic, target, env=(100, 100)):
    env_w, env_h = env
    epc = np.array(episodic)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect(1)
    major_locator = MultipleLocator(2)
    ax.xaxis.set_major_locator(major_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.grid(linewidth=2)
    ln, = ax.plot([], [], 'r-', animated=False)

    def init():
        ax.set_xlim(0, env_w)
        ax.set_ylim(0, env_h)
        ax.scatter(target[0], target[1], marker='*', s=100, color="b")
        return ln

    def update(n, length=50):
        n_ref = n - length
        if n_ref < 0: n_ref = 0
        xdata, ydata = epc[n_ref:n, 0], epc[n_ref:n, 1]
        ln.set_data(xdata, ydata) 
        return ln

    ani = FuncAnimation(fig, update, frames=3000,
                        init_func=init,interval=1)
    plt.show()
