from functools import partial
from typing import Tuple, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import xarray as xr
from src.nn import HealPIXUNet
from src.schedules import ContinuousVESchedule
from src.samplers import ContinuousHeunSampler



################################################################################
#                               BATCHES PROCESSING                             #
################################################################################

@eqx.filter_jit
def normalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Normalize data using mean and standard deviation."""
    return (x - μ) / σ

@eqx.filter_jit
def denormalize(x: jnp.ndarray, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Denormalize data using mean and standard deviation."""
    return σ * x + μ

@eqx.filter_jit
def process_single(pattern: jnp.ndarray, samples: jnp.ndarray,  μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Process a single sample for model input."""
    x = jnp.concatenate([samples, pattern[None, ...]], axis=0)
    return normalize(x, μ, σ)

@eqx.filter_jit
def process_batch(batch: Tuple, μ: jnp.ndarray, σ: jnp.ndarray) -> jnp.ndarray:
    """Process a batch of samples."""
    patterns, samples = batch
    return jax.vmap(partial(process_single, μ=μ, σ=σ))(patterns, samples)



################################################################################
#                               SAMPLING FUNCTIONS                             #
################################################################################

def create_sampler(model: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                   μ: jnp.ndarray, σ: jnp.ndarray) -> ContinuousHeunSampler:
    """Create a sampler for a given pattern."""
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def model_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return model(x, t)
    return ContinuousHeunSampler(schedule, model_with_context, (4, 96, 192))

@eqx.filter_jit
def draw_samples_single(model: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                        n_samples: int, n_steps: int, μ: jnp.ndarray, σ: jnp.ndarray,
                        key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a given pattern."""
    sampler = create_sampler(model, schedule, pattern, μ, σ)
    samples = sampler.sample(n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])

@eqx.filter_jit
def draw_samples_batch(model: eqx.Module, schedule: Any, pattern_batch: jnp.ndarray,
                      n_samples: int, n_steps: int, μ: jnp.ndarray,
                      σ: jnp.ndarray, key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a batch of patterns."""
    keys = jr.split(key, pattern_batch.shape[0])
    Γ = partial(draw_samples_single,
                model=model,
                schedule=schedule,
                n_samples=n_samples,
                n_steps=n_steps, μ=μ, σ=σ)
    return jax.vmap(Γ)(pattern=pattern_batch, key=keys)




################################################################################
#                               DEMO VARIABLES                                 #
################################################################################
β = jnp.load("cache/β.npy")
stats = jnp.load("cache/μ_σ.npz")
μ_train, σ_train = jnp.array(stats['μ']), jnp.array(stats['σ'])
σmax = float(jnp.load("cache/σmax.npy"))
edges_data = jnp.load("cache/edges.npz")
to_healpix = jnp.array(edges_data['to_healpix']).astype(jnp.int32)
to_latlon = jnp.array(edges_data['to_latlon']).astype(jnp.int32)
lat = jnp.load("cache/lat.npy")
lon = jnp.load("cache/lon.npy")
varnames = ['tas', 'pr', 'hurs', 'sfcWind']

nn = HealPIXUNet(
    input_size=(5, 96, 192),
    nside=64,
    enc_filters=[32, 64, 128, 256, 512],
    dec_filters=[256, 128, 64, 32, 32],
    out_channels=4,
    temb_dim=256,
    healpix_emb_dim=5,
    edges_to_healpix=to_healpix,
    edges_to_latlon=to_latlon
)
nn = eqx.tree_deserialise_leaves("cache/ckpt.eqx", nn)

@jax.jit
def compiled_nn(x, t):
    return nn(x, t)
x = jnp.zeros((5, 96, 192))
_ = compiled_nn(x, jnp.zeros((1,)))

schedule = ContinuousVESchedule(0.01, σmax)


def make_emulator(n_steps=30):
    emulator_from_pattern = partial(draw_samples_single,
                                    model=compiled_nn,
                                    schedule=schedule,
                                    n_samples=1,
                                    μ=μ_train,
                                    σ=σ_train)
    # dry run to compile the function
    _ = emulator_from_pattern(pattern=jnp.zeros((96, 192)), n_steps=n_steps, n_samples=1, key=jr.PRNGKey(0))
    def emulator(ΔT, month, seed):
        pattern = β[month - 1, :, 1] * ΔT + β[month - 1, :, 0]
        pattern = pattern.reshape((96, 192))
        key = jr.PRNGKey(seed)
        samples = emulator_from_pattern(pattern=pattern, n_steps=n_steps, key=key)
        return samples
    return emulator


# ds = xr.Dataset(
#     {
#         var: (('member', 'lat', 'lon'), samples[:, i, :, :])
#         for i, var in enumerate(varnames)
#     },
#     coords={
#         'member': jnp.arange(n_samples) + 1,
#         'lat': lat,
#         'lon': lon,
#     }
# )
# return ds