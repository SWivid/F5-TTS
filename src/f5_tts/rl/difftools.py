import abc
import random

import torch
from torchdiffeq._impl.event_handling import find_event
from torchdiffeq._impl.misc import Perturb, _check_inputs, _flat_to_shape, _handle_unused_kwargs


class EulerSolver(metaclass=abc.ABCMeta):
    order = 1

    def __init__(
        self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, **unused_kwargs
    ):
        self.atol = unused_kwargs.pop("atol")
        unused_kwargs.pop("rtol", None)
        unused_kwargs.pop("norm", None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {"callback_step"}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    def _step_func(self, func, t0, dt, t1, y0):
        f0, mu, log_sig = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0, mu, log_sig

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        pro_result = []

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            # Upstream F5R behavior: randomly skip gradient tracking on some steps for speed.
            if_no_grad = random.uniform(0, 1)
            if if_no_grad > 0.05 and len(pro_result) > 1:
                with torch.no_grad():
                    dy, f0, mu, log_sig = self._step_func(self.func, t0, dt, t1, y0)
                    y1 = y0 + dy
                pro_result.append([f0, mu, log_sig, False])
            else:
                dy, f0, mu, log_sig = self._step_func(self.func, t0, dt, t1, y0)
                y1 = y0 + dy
                pro_result.append([f0, mu, log_sig, True])

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1, mu, log_sig = self.func(t1, y1)
                    pro_result.append([f1, mu, log_sig])
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution, pro_result

    def integrate_until_event(self, t0, event_fn):
        warn_text = "Event handling for fixed step solvers currently requires `step_size` to be provided in options."
        assert self.step_size is not None, warn_text

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        pro_result = []
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0, mu, log_sig = self._step_func(self.func, t0, dt, t1, y0)
            pro_result.append([mu, log_sig])
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":

                    def interp_fn(self, t):
                        return self._linear_interp(t0, t1, y0, y1, t)

                elif self.interp == "cubic":
                    f1, mu, log_sig = self.func(t1, y1)
                    pro_result.append([mu, log_sig])

                    def interp_fn(self, t):
                        return self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)

                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution, pro_result

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = t1 - t0
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


SOLVERS = {
    "euler": EulerSolver,
}


def odeint_rl(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(
        func, y0, t, rtol, atol, method, options, event_fn, SOLVERS
    )

    solver = EulerSolver(func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution, pro_result = solver.integrate(t)
    else:
        event_t, solution, pro_result = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution, pro_result
    return event_t, solution, pro_result
