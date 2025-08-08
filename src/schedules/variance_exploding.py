import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class ContinuousVESchedule(eqx.Module):
    σmin: float
    σmax: float
    logσmin: float
    logσmax: float
    σd: float
    timesteps: jnp.ndarray
    def __init__(self, σmin, σmax, timesteps=None):
        self.σmin = σmin
        self.σmax = σmax
        self.logσmin = jnp.log(σmin)
        self.logσmax = jnp.log(σmax)
        self.σd = σmax / σmin
        self.timesteps = timesteps

    @eqx.filter_jit
    def σ(self, t):
        return self.σmin * jnp.sqrt(self.σd ** (2 * t) - 1)
    
    @eqx.filter_jit
    def t(self, σ):
        return jnp.log1p((σ / self.σmin) ** 2) / (2 * jnp.log(self.σd))

    @eqx.filter_jit
    def g2(self, t):
        dσ2dt = (
            (self.σmin ** 2)
            * (self.σmax / self.σmin) ** (2 * t)
            * (2 * jnp.log(self.σmax / self.σmin))
        )
        return dσ2dt

    @eqx.filter_jit
    def sample_σ(self, key):
        logσ = jr.uniform(key, minval=self.logσmin, maxval=self.logσmax)
        return jnp.exp(logσ)

    def get_timesteps(self, steps):
        tmin = jnp.log(2) / (2 * jnp.log(self.σd))
        tmax = 1.0
        if self.timesteps is not None:
            timesteps = self.timesteps
            timesteps = timesteps.at[0].set(tmin)
            timesteps = timesteps.at[-1].set(tmax)
        else:
            timesteps = jnp.linspace(tmin, tmax, steps)
        return timesteps