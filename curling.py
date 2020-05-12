# The curling should in [r, env_w-r] * [r, env_h-r].
# The action is taken at 0.1 second and 
# can be changed every 0.1 second.


import math
import random

from utils import clamp, is_illegal
from utils import plot, solve_quad_eq


eps = 1e-3

class Curling(object):
    def __init__(self, env=(100, 100), vd_max=10.0):
        self._env_w, self._env_h = env 
        self._ref_rates = 0.9
        self._air_res_r = -5e-3

        self._r = 1
        self._target_x = random.random() * (self._env_h 
            - 2 * self._r) + self._r
        self._target_y = random.random() * (self._env_w 
            - 2 * self._r) + self._r
        self._m = 1.
        self._x = random.random() * (self._env_h - 2 * self._r) + self._r
        self._y = random.random() * (self._env_w - 2 * self._r) + self._r
        self._vx = random.random() * vd_max * 2 - vd_max
        self._vy = random.random() * vd_max * 2 - vd_max

        self.Fx = 0.
        self.Fy = 0.

        self.cnt = 0

    @property   
    def target(self):
        return (self._target_x, self._target_y)

    @property   
    def position(self):
        return (self._x, self._y)

    @property   
    def state(self):
        return (self.position, self.target, (self._vx, self._vy))

    @property
    def v_theta(self):
        return math.atan2(self._vy, self._vx)

    def move(self, duration=0.01):
        air_res = (self._vx * self._vx +
            self._vy * self._vy) * self._air_res_r
        ax = (((air_res * math.cos(self.v_theta))
                     + self.Fx)) / self._m
        ay = (((air_res * math.sin(self.v_theta))
                     + self.Fy)) / self._m
        dx = self._vx * duration + (ax *
                     duration * duration) / 2.0
        dy = self._vy * duration + (ay *
                     duration * duration) / 2.0

        new_x = self._x + dx
        new_y = self._y + dy

        illegal = is_illegal((new_x, new_y), self._r, 
            (self._env_w, self._env_h), eps)

        if not illegal:

            self._x, self._y = clamp((new_x, new_y), 
                self._r, (self._env_w, self._env_h))
            self._vx += ax * duration
            self._vy += ay * duration

        elif illegal == 1:  #x < r

            bt = solve_quad_eq(ax/2.0, 
                self._vx, self._r-self._x)

            self._x = self._r
            self._y += self._vy * bt + (ay *
                                  bt * bt) / 2.0
            self._vx += ax * bt
            self._vx = -self._vx
            self._vy += ay * bt

            self._reflect_update(duration-bt, illegal)

        elif illegal == 2:  #y < r

            bt = solve_quad_eq(ay/2.0, 
                self._vy, self._r-self._y)
            self._x += self._vx * bt + (ax *
                                  bt * bt) / 2.0
            self._y = self._r
            self._vx += ax * bt
            self._vy += ay * bt
            self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)

        elif illegal == 4:  #x > env_w - r

            bt = solve_quad_eq(ax/2.0, 
                self._vx, self._env_w-self._r-self._x)
            self._x = self._env_w - self._r
            self._y += self._vy * bt + (ay *
                                  bt * bt) / 2.0

            self._vx += ax * bt
            self._vx = -self._vx
            self._vy += ay * bt

            self._reflect_update(duration-bt, illegal)


        elif illegal == 8:  #y > env_h - r
            
            bt = solve_quad_eq(ay/2.0, 
                self._vy, self._env_h-self._r-self._y)
            self._x += self._vx * bt + (ax *
                                  bt * bt) / 2.0
            self._y = self._env_h - self._r
            self._vx += ax * bt
            self._vy += ay * bt
            self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)
        elif illegal == 3:  #x < r and y < r

            btx = solve_quad_eq(ax/2.0, 
                self._vx, self._r-self._x)
            bty = solve_quad_eq(ay/2.0, 
                self._vy, self._r-self._y)
            bt = btx

            if btx <= bty:
                illegal = 1
                self._x = self._r
                self._y += self._vy * bt + (ay *
                                      bt * bt) / 2.0
                self._vx += ax * bt
                self._vx = -self._vx
                self._vy += ay * bt
            else:
                bt = bty
                illegal = 2
                self._x += self._vx * bt + (ax *
                                      bt * bt) / 2.0
                self._y = self._r
                self._vx += ax * bt
                self._vy += ay * bt
                self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)

        elif illegal == 9:  #x < r and y > env_h - r

            btx = solve_quad_eq(ax/2.0, 
                self._vx, self._r-self._x)
            bty = solve_quad_eq(ay/2.0, 
                self._vy, self._env_h-self._r-self._y)
            bt = btx

            if btx <= bty:
                illegal = 1
                self._x = self._r
                self._y += self._vy * bt + (ay *
                                      bt * bt) / 2.0
                self._vx += ax * bt
                self._vx = -self._vx
                self._vy += ay * bt
            else:
                bt = bty
                illegal = 2
                self._x += self._vx * bt + (ax *
                                      bt * bt) / 2.0
                self._y = self._env_h - self._r
                self._vx += ax * bt
                self._vy += ay * bt
                self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)

        elif illegal == 6:  #x > env_w - r and y < r
            btx = solve_quad_eq(ax/2.0, 
                self._vx, self._env_w-self._r-self._x)
            bty = solve_quad_eq(ay/2.0, 
                self._vy, self._r-self._y)
            bt = btx

            if btx <= bty:
                illegal = 1
                self._x = self._env_w-self._r
                self._y += self._vy * bt + (ay *
                                      bt * bt) / 2.0
                self._vx += ax * bt
                self._vx = -self._vx
                self._vy += ay * bt
            else:
                bt = bty
                illegal = 2
                self._x += self._vx * bt + (ax *
                                      bt * bt) / 2.0
                self._y = self._r
                self._vx += ax * bt
                self._vy += ay * bt
                self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)


        elif illegal == 12:  #x > env_w - r and y > env_h - r
            
            btx = solve_quad_eq(ax/2.0, 
                self._vx, self._env_w-self._r-self._x)
            bty = solve_quad_eq(ay/2.0, 
                self._vy, self._env_h-self._r-self._y)
            bt = btx

            if btx <= bty:
                illegal = 1
                self._x = self._env_w-self._r
                self._y += self._vy * bt + (ay *
                                      bt * bt) / 2.0
                self._vx += ax * bt
                self._vx = -self._vx
                self._vy += ay * bt
            else:
                bt = bty 
                illegal = 2
                self._x += self._vx * bt + (ax *
                                      bt * bt) / 2.0
                self._y = self._env_h-self._r
                self._vx += ax * bt
                self._vy += ay * bt
                self._vy = -self._vy

            self._reflect_update(duration-bt, illegal)

    def action(self, Fx, Fy):
        self.Fx, self.Fy = Fx, Fy

    def reset(self, pos, v, target=None):
        self._x, self._y = pos
        self._vx, self._vy = v
        if target:
            self._target_x, self._target_y = target

    def reward(self, pos=None):
        x, y = pos if pos else self.position
        tx, ty = self.target
        dx = x - tx
        dy = y - ty
        d = math.sqrt((dx * dx + dy * dy))
        return -d

    def _reflect_update(self, duration, illegal):
        
        # mid_v = math.sqrt(self._vx * self._vx +
        #     self._vy * self._vy) * self._ref_rates
        # self._vx = mid_v * math.cos(self.v_theta)
        # self._vy = mid_v * math.cos(self.v_theta)
        
        if illegal&1:
            self._vx *= self._ref_rates
        else:
            self._vy *= self._ref_rates

        self.move(duration=duration)


def main():
    action = [(5, 5), (-5, 5), (5, -5), (-5, -5)]
    curling = Curling()
    episodic = [curling.position]
    from utils import default_action
    for step in range(300):
        # if step % 300 == 0:
        #     curling = Curling()
        a = default_action(curling.position,
            curling.target)
        curling.action(*action[a]) #random.randint(0, 3)
        for interval in range(10):
            curling.cnt += 1
            curling.move(0.01)
        episodic.append(curling.position)
    
    plot(episodic, curling.target)


if __name__ == '__main__':
    main()