import jax
import jax.numpy as jnp
import math
import equinox as eqx
import functools as ft
import jax.random as jr
import diffrax as dfx


class ContinuousEulerSampler:
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
                              solver=dfx.Euler(),
                              t0=jnp.max(timesteps),
                              t1=jnp.min(timesteps),
                              stepsize_controller=solversteps,
                              dt0=None,
                              y0=x0)
        return sol.ys

    @eqx.filter_jit
    def sample(self, N, key=jr.PRNGKey(0), steps=300):
        keys = jax.random.split(key, N)
        x0 = jr.normal(keys[0], (N, math.prod(self.data_shape)), dtype=jnp.float16) * self.schedule.σmax
        reverse_timesteps = self.schedule.get_timesteps(steps)[::-1]
        sampler = ft.partial(self.precursor_solver, reverse_timesteps)
        samples = jax.vmap(sampler)(x0)
        samples = jnp.reshape(samples, (N, *self.data_shape))
        return samples
