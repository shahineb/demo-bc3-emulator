import jax
import jax.numpy as jnp
import math
import equinox as eqx
import functools as ft
import jax.random as jr
import diffrax as dfx


class ContinuousHeunSampler:
    def __init__(self, schedule, model, data_shape):
        @eqx.filter_jit
        def denoiser_precursor(model, data_shape, x, σ):
            xr = jnp.reshape(x, data_shape)
            denoised_x = model(xr, σ)
            return jnp.reshape(denoised_x, (math.prod(data_shape)))

        @eqx.filter_jit
        def drift_precursor(denoiser, schedule, x, t):
            g2 = schedule.g2(t)
            σ = schedule.σ(t)
            scaling = 1 + σ
            prefactor = g2 / (2 * σ**2)
            scaled_score = denoiser(x / scaling, σ) - x
            drift = - prefactor * scaled_score
            return drift

        denoiser = ft.partial(denoiser_precursor, model, data_shape)
        drift = ft.partial(drift_precursor, denoiser, schedule)
        self.schedule = schedule
        self.data_shape = data_shape
        self.drift = drift
    
    @eqx.filter_jit
    def precursor_solver(self, timesteps, x0):
        def drift_diffrax_signature(t, y, args=None):
            return self.drift(y, t)
        f = dfx.ODETerm(drift_diffrax_signature)
        solversteps = dfx.StepTo(timesteps)
        sol = dfx.diffeqsolve(terms=f,
                              solver=dfx.Heun(),
                              t0=jnp.max(timesteps),
                              t1=jnp.min(timesteps),
                              stepsize_controller=solversteps,
                              dt0=None,
                              y0=x0)
        return sol.ys

    @eqx.filter_jit
    def sample(self, key=jr.PRNGKey(0), steps=30):
        x0 = jr.normal(key, (math.prod(self.data_shape),)) * self.schedule.σmax
        reverse_timesteps = self.schedule.get_timesteps(steps)[::-1]
        sample = self.precursor_solver(reverse_timesteps, x0)
        sample = jnp.reshape(sample, self.data_shape)
        return sample
