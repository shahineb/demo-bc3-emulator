from functools import partial
from typing import Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import xarray as xr
from src.nn import HealPIXUNet
from src.schedules import ContinuousVESchedule
from src.samplers import ContinuousEulerSampler



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
                   μ: jnp.ndarray, σ: jnp.ndarray) -> ContinuousEulerSampler:
    """Create a sampler for a given pattern."""
    context = normalize(pattern, μ[-1], σ[-1])[None, ...]
    def model_with_context(x, t):
        x = jnp.concatenate((x, context), axis=0)
        return model(x, t)
    return ContinuousEulerSampler(schedule, model_with_context, (4, 96, 192))

@eqx.filter_jit
def draw_samples_single(model: eqx.Module, schedule: Any, pattern: jnp.ndarray,
                        n_samples: int, n_steps: int, μ: jnp.ndarray, σ: jnp.ndarray,
                        key: jr.PRNGKey = jr.PRNGKey(0)) -> jnp.ndarray:
    """Draw samples for a given pattern."""
    sampler = create_sampler(model, schedule, pattern, μ, σ)
    samples = sampler.sample(N=n_samples, steps=n_steps, key=key)
    return denormalize(samples, μ[:-1], σ[:-1])



################################################################################
#                             CLIMATE SCENARIO                                 #
################################################################################

# kranke - August 2025
# Script to define the climateScenario class for handling emissions scenarios
# From https://github.com/kranke-git/em2gmst/tree/main


class climateScenario:
    
    def __init__( self, emissions, name = None ):
        """
        This Python function initializes an object with emissions data and an optional name attribute.
        
        :param emissions: The `emissions` parameter can be either a string (for presets) or a pandas DataFrame (for custom emissions data)
        :param name: The `name` parameter in the `__init__` method is an optional parameter for the custom emissions data
        """
        if isinstance( emissions, str ):
            self.name        = name or emissions
            self.preset_name = emissions
            self.datadir = 'https://svante.mit.edu/~pgiani/emissionsdata'
            self.emissions   = self._load_preset( emissions )
        elif isinstance( emissions, pd.Series ):
            self.name        = name or "Custom"
            self.preset_name = None
            self.emissions   = pd.DataFrame( emissions.sort_index() )
            self.emissions['ppmCO2'] = self.emissions['CO2emissions_GtC'] / 2.12  # Convert GtC to ppm for the carbon cycle model
            self.flagEmissions = True
        else:
            raise ValueError("emissions must be a string or a pandas Series.")

        
    def _load_preset( self, name ):
        """Load emissions data from a CSV file for known presets."""
        name = name.lower()

        if name == "historical":
            return self._load_historical()
        elif name.startswith("ssp"):
            return self._load_sspScenario()
        elif name == 'pulseco2':
            return self._load_pulseCO2()
        elif name == 'abrupt-4xco2':
            return self._load_abrupt4xCO2()
        elif name.startswith("cmip7_"):
            return self._load_cmip7scenario()
        else:
            raise ValueError(f"Unknown preset '{name}'")
    
    def _load_pulseCO2( self ):
        """Load pulse CO2 emissions data."""
        # Pulse CO2 scenario: 100 GtCO2 pulse in 1750
        years = np.arange( 1750, 2001 )
        df = pd.DataFrame( index = years, data = 0.0, columns = ['ppmCO2'] )
        df.loc[ 1750, 'ppmCO2']  = 100.0 / 2.12
        self.flagEmissions       = True
        return df
    
    def _load_abrupt4xCO2( self ):
        """Load abrupt 4xCO2 concentration data."""
        # Abrupt 4xCO2 scenario: 4x pre-industrial CO2 concentration in 1750
        years = np.arange( 1750, 2001 )
        df    = pd.DataFrame( index = years, data = 0.0, columns = ['Catm'] )
        df[ 'Catm' ]       = 277.15 * 4.0
        self.flagEmissions = False
        return df

    
    def _load_historical( self ):
        """Load historical emissions data from a CSV file."""
        filename     = self.datadir + '/historicalEmissions.csv'
        df           = pd.read_csv( filename, index_col = 0 ) # This values are in GtCO2
        df['ppmCO2'] = ( df['FF'] + df['LU'] ) / 2.12         #  ppm/year; for the carbon cycle model
        self.flagEmissions = True
        return df 
    
    def _load_sspScenario( self ):
        """Load SSP Concentrations data from a CSV file."""
        histfile = self.datadir + '/CO2molefractions_hist_ssp/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.csv'
        sspfiles = {'ssp119': 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-IMAGE-ssp119-1-2-1_gr1-GMNHSH_2015-2500.csv',
                    'ssp126': 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_2015-2500.csv',
                    'ssp245': 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.csv',
                    'ssp370': 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.csv',
                    'ssp585': 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-1_gr1-GMNHSH_2015-2500.csv' }
        if self.name in sspfiles.keys():
            sspfile  = f"{self.datadir}/CO2molefractions_hist_ssp/{sspfiles[ self.name ] }"
        else:
            raise ValueError(f"Unknown SSP scenario '{self.name}'. Valid options are: {', '.join(sspfiles.keys())}")    
        # Process historical and SSP concentration data
        histdf   = pd.read_csv( histfile )
        sspdf    = pd.read_csv( sspfile )
        merged   = pd.concat( [ histdf, sspdf ] ).sort_values(by='year').drop_duplicates( subset='year', keep='last').sort_values('year')
        merged.set_index( 'year', inplace = True )
        merged   = merged.rename( columns = { 'data_mean_global': 'Catm' } ) 
        syear    = 1750
        eyear    = 2100
        # Set emission flag to be false for SSP scenarios
        self.flagEmissions = False
        return( merged.loc[ syear:eyear ]['Catm'] )
    
    def _load_cmip7scenario( self ):
        """Load CMIP7 Emissions data from the big CSV file."""
        filename = self.datadir + '/cmip7_extensions_1750-2500.csv'
        # Read in the data and subset for the correct scenario and CO2 only
        df             = pd.read_csv( filename, index_col = 0 )
        dfCO2          = df[ ( ( df['variable'] == 'CO2 AFOLU' ) | ( df['variable'] == 'CO2 FFI' ) ) ]
        scen_name_file = self.name.replace( 'cmip7_', '', 1 )
        if scen_name_file not in dfCO2.index:
            raise ValueError(f"Unknown CMIP7 scenario '{self.name}'. Valid options are: {', '.join( dfCO2.index.unique() ) }" )
        else:
            dfscen         = dfCO2[ dfCO2.index == scen_name_file ]
        # Convert to a DataFrame with 'year' as index
        numeric_data   = dfscen.drop( columns=['region', 'variable', 'ar6_gwp_mass_adjusted', 'unit'] )
        sum_by_year    = numeric_data.sum( axis=0 )
        df_sum           = sum_by_year.reset_index()
        df_sum.columns   = ['year', 'total_GtCO2']
        df_sum[ 'year' ] = ( df_sum[ 'year' ].astype( float ) - 0.5).astype( int )
        df_sum['ppmCO2'] = df_sum['total_GtCO2'] / 2.12 * 12/44  # Convert GtCO2 to ppm for the carbon cycle model
        df_sum.set_index( 'year', inplace=True )
        # These are emission driven scenarios, so we set the flag accordingly
        self.flagEmissions = True
        return df_sum

        
    def integrate( self, dt = 0.1 ):
        """
        This function integrates emissions data over time to simulate the carbon cycle and update atmospheric CO2 concentration and GMST.
        :param self: The `self` parameter refers to the instance of the class in which this method is defined
        :param dt: The `dt` parameter in the `integrate` method represents the time step for the integration process, which is set to 0.1 by default
        """
        # Unpack variables from self
        emissions = self.emissions
        # Assign time stuff and initialize variables
        startTime          = emissions.index.min()
        endTime            = emissions.index.max()
        tvec               = np.arange( startTime, endTime + dt, dt )
        Catm               = np.full( int( ( endTime - startTime ) / dt ) + 1, np.nan )
        RFs                = Catm.copy()
        gmst               = Catm.copy()
        oceantemps         = np.full( ( len( tvec ), 4 ), np.nan )
        Catm[ 0 ]          = 277.15 # Pre-industrial CO2 concentration [ppm]
        RFs[ 0 ]           = 0.0    # Initial Radiative Forcing [W/m^2]
        gmst[ 0 ]          = 0.0    # Initial GMST anomaly [K]
        oceantemps[ 0, : ] = 0.0 # Initial ocean temperature anomalies [K]
        cpool              = np.zeros( ( 1, 4 ) )
        # Main loop over time #
        idx = 0
        for currtime in tvec[:-1]:
            # Update the current time index
            idx += 1
            # Carbon Cycle (skip it if we have concentrations, otherwise run it)
            if self.flagEmissions == True:
                cpool       = self._carbonCycle( cpool, emissions, currtime, dt )
                Catm[ idx ] = Catm[ 0 ] + np.sum( cpool )
            else:
                Catm[ idx ] = emissions.iloc[ int( np.searchsorted( emissions.index.values, currtime, side = 'right' ) - 1 ) ]
            # RF Model
            RFs[ idx ]  = self._radiativeForcing( Catm[ idx ] )
            # Climate Model
            gmst[ idx ], oceantemps[ idx, : ] = self._climateModel( RFs[ idx - 1 ], gmst[ idx - 1], oceantemps[ idx-1, : ], dt )
        # Output stuff #
        indices     = np.searchsorted( tvec, emissions.index, side='right') - 1
        Catm_interp = Catm[ indices ]
        RFs_interp  = RFs[ indices ]
        gmst_interp = gmst[ indices ]
        if self.flagEmissions == True:
            cumulative_emissions = emissions['ppmCO2'].cumsum() * 2.12  # Convert ppm to GtC
            self.outdf  = pd.DataFrame({'emissionsGtC': emissions['ppmCO2']*2.12,
                                        'cumulativeEmissionsGtC': cumulative_emissions,
                                        'Catm': Catm_interp, 
                                        'RF': RFs_interp, 
                                        'GMST': gmst_interp,
                                        }, index = emissions.index )
        else:
            self.outdf  = pd.DataFrame({'Catm': Catm_interp, 
                                        'RF': RFs_interp, 
                                        'GMST': gmst_interp,
                                        }, index = emissions.index )
        
    def _carbonCycle( self, cpool, emissions, currtime, dt ):
        """
        This function simulates the carbon cycle by updating the atmospheric CO2 concentration based on emissions data.
        :param self: The `self` parameter refers to the instance of the class in which this method is defined
        """
        
        # Carbon cycle parameters (from Joos et al., 2013)
        a   = np.array( [ 0.2173, 0.2240, 0.2824, 0.2763 ] ) # [ fractions in each reservoir ]
        tau = np.array( [ 1e6, 394.4, 36.54, 4.304 ] )       # [ time scale in years for each reservoir ]

        # Unpack variables from self
        Evals = emissions['ppmCO2'].values
        tvec  = emissions.index.values
        idx   = np.searchsorted( tvec, currtime + dt/10, side = 'right' ) - 1 # Figure out where we are in time
        Et    = Evals[ idx ] if idx >= 0 else 0

        # Vectorized update of all carbon reservoirs
        dydt  = a * Et - cpool / tau
        cpool = cpool + dt * dydt

        # Sum it up to get C concentrations and assign to outstruct
        return cpool
    
    def _radiativeForcing( self, C ):
        """
        Compute Radiative Forcing (RF) from CO2 concentration C (ppm), based on AR6 Table 6.2 parameters (similar to EnROADS).
        Parameters:
        - C : scalar or numpy array of CO2 concentrations [ppm]
        Returns:
        - RF : Radiative Forcing [W/m^2], same shape as C
        """

        a1 = -2.4785e-7   # W m-2 ppm-2
        b1 = 7.5906e-4    # W m-2 ppm-1
        d1 = 5.2488       # W m-2
        C0 = 277.15       # Preindustrial Concentration [ppm]

        # Calculate Camax based on the formula provided
        Camax = C0 - b1 / (2 * a1)

        # Ensure C is a numpy array for vectorized logic
        C              = np.array( C, dtype = float )
        coefC          = np.full_like( C, d1 )
        mask1          = C >= Camax
        mask2          = (C < Camax) & (C > C0)
        mask3          = C <= C0 
        coefC[ mask1 ] = d1 - (b1 ** 2) / (d1 * a1)
        coefC[ mask2 ] = d1 + a1 * (C[mask2] - C0) ** 2 + b1 * (C[mask2] - C0)
        coefC[ mask3 ] = d1
        rf             = coefC * np.log( C / C0 )
        return( rf )

    def _climateModel( self, RF, tanm_ao, tanm_d, dt ):
        """_summary_

        Args:
            cpool (_type_): _description_
            emissions (_type_): _description_
            tvec (_type_): _description_
            currtime (_type_): _description_
            dt (_type_): _description_
        """
        # Define parameters
        eartharea = 5.1e14               # m2
        lfrac     = 0.292                # -
        lthk      = 8.4                  # m
        othk      = 100                  # m
        omhc      = 4186                 # J kg-1 K-1
        odens     = 1000                 # kg m-3
        dthk      = np.array([300, 300, 1300, 1800])  # m
        spy       = 3600 * 24 * 365.25   # s yr-1
        ecs       = 3.0                  # W m-2
        ohtr      = 1.23                 # W m-2 K-1

        # Initialization
        NLEV          = len( dthk )
        ao_vol        = eartharea * ( lfrac * lthk + (1 - lfrac) * othk )
        ao_vhc        = omhc * odens
        ao_hc         =  (ao_vol * ao_vhc ) / eartharea
        d_vol         = ( 1 - lfrac ) * eartharea * dthk
        d_vhc         = ao_vhc
        d_hc          = ( d_vol * d_vhc ) / eartharea
        layert        = np.concatenate( ( [ othk ], dthk ) )
        mlayert       = np.mean( np.vstack( ( layert[:-1], layert[1:] ) ), axis=0 )
        htc           = ohtr * mlayert[ 0 ]
        feedbackParam = self._radiativeForcing( 277.15 * 2 ) / ecs

        # Initial condition for current step
        aoQ = tanm_ao * ao_hc
        dQ  = tanm_d  * d_hc

        # Terms and update
        feedcooling = tanm_ao * feedbackParam
        htr         = np.full( NLEV, np.nan )
        htr[0]      = ( tanm_ao - tanm_d[0] ) * htc / mlayert[ 0 ]
        htr[1:]     = ( ( dQ[:-1] / d_hc[:-1]) - ( dQ[1:] / d_hc[1:] ) ) * htc / mlayert[1:]
        aoQnew      = aoQ + dt * ( RF - feedcooling - htr[0] ) * spy # Update equation for gmst
        dQnew       = np.full( NLEV, np.nan )
        dQnew[:-1]  = dQ[:-1] + dt * ( htr[:-1] - htr[1:] ) * spy
        dQnew[-1]   = dQ[-1]  + dt * htr[-1] * spy

        # Post-processing
        return aoQnew / ao_hc, dQnew / d_hc
    
    def plotOutput(self, year=None):
        """
        This function plots the output of the climate scenario, including CO2 concentration, emissions, radiative forcing, and GMST.
        """
        df = self.outdf
        df = df.loc[df.index <= 2150]
        df = df.loc[df.index >= 1850]
        def set_three_ticks(ax, values, selected):
            vmin, vmax = np.ceil(np.min(values)), np.floor(np.max(values))
            ticks = sorted({vmin, selected, vmax})
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:.0f}" if t.is_integer() else f"{t:.1f}" for t in ticks])

        
        if self.flagEmissions:
        
            fig, axes = plt.subplots( 1, 3, figsize = (15, 3.5), sharex=True)

            # --- Panel 1 ---
            ax1 = axes[0]
            ax2 = ax1.twinx()
            ax1.plot(df.index, df["Catm"], color="tab:blue")
            ax1.set_ylabel("CO₂ Concentration (ppm)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid( True, which='both', linestyle='--', alpha=0.0)
            ax2.plot(df.index, df["emissionsGtC"], color = "tab:red" )
            ax2.set_ylabel("Emissions (GtC)", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax1.set_title("CO₂ Emissions vs Concentrations")
            if year:
                y_emissions = df.loc[year, "emissionsGtC"]
                y_catm = df.loc[year, "Catm"]

                ax1.plot(year, y_catm, marker="o", ms=6, color="tab:blue")
                ax1.plot([year, year], [ax1.get_ylim()[0], y_catm],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax1.plot([ax1.get_xlim()[0], year], [y_catm, y_catm],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:blue")
                set_three_ticks(ax1, df["Catm"], y_catm)

                ax2.plot(year, y_emissions, marker="o", ms=6, color="tab:red")
                ax2.plot([year, year], [ax2.get_ylim()[0], y_emissions],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax2.plot([year, ax1.get_xlim()[-1]], [y_emissions, y_emissions],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:red")
                set_three_ticks(ax2, df["emissionsGtC"], y_emissions)

            # --- Panel 2 ---
            ax1 = axes[1]
            ax2 = ax1.twinx()

            ax1.plot(df.index, df["Catm"], color="tab:blue")
            ax1.set_ylabel("CO₂ Concentration (ppm)", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid( True, which='both', linestyle='--', alpha=0.0)  # Grid on left axis
            ax2.plot(df.index, df["RF"], color="tab:green")
            ax2.set_ylabel("Effective Radiative Forcing (W m⁻²)", color="tab:green")
            ax2.tick_params(axis="y", labelcolor="tab:green")
            ax1.set_title("CO₂ concentrations vs Radiative Forcing")
            if year:
                y_catm = df.loc[year, "Catm"]
                y_rf   = df.loc[year, "RF"]

                ax1.plot(year, y_catm, marker="o", ms=6, color="tab:blue")
                ax1.plot([year, year], [ax1.get_ylim()[0], y_catm],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax1.plot([ax1.get_xlim()[0], year], [y_catm, y_catm],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:blue")
                set_three_ticks(ax1, df["Catm"], y_catm)
                
                ax2.plot(year, y_rf, marker="o", ms=6, color="tab:green")
                ax2.plot([year, year], [ax2.get_ylim()[0], y_rf],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax2.plot([year, ax1.get_xlim()[-1]], [y_rf, y_rf],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:green")
                set_three_ticks(ax2, df["RF"], y_rf)
                

            # --- Panel 3 ---
            ax1 = axes[2]
            ax2 = ax1.twinx()
            ax1.plot( df.index, df['cumulativeEmissionsGtC'], color="tab:purple" )
            ax1.set_ylabel("Cumulative Emissions (GtC)", color="tab:purple")
            ax1.tick_params(axis="y", labelcolor="tab:purple")
            ax1.grid( True, which='both', linestyle='--', alpha=0.0)  # Grid on left axis
            ax2.plot( df.index, df["GMST"], color="tab:orange")
            ax2.set_ylabel("Temp Anomaly (K)", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax1.set_title("Cumulative Emissions vs GMST response")
            if year:
                y_cumulative_emissions = df.loc[year, "cumulativeEmissionsGtC"]
                y_gmst = df.loc[year, "GMST"]

                ax1.plot(year, y_cumulative_emissions, marker="o", ms=6, color="tab:purple")
                ax1.plot([year, year], [ax1.get_ylim()[0], y_cumulative_emissions],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax1.plot([ax1.get_xlim()[0], year], [y_cumulative_emissions, y_cumulative_emissions],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:purple")
                set_three_ticks(ax1, df["cumulativeEmissionsGtC"], y_cumulative_emissions)

                ax2.plot(year, y_gmst, marker="o", ms=6, color="tab:orange")
                ax2.plot([year, year], [ax2.get_ylim()[0], y_gmst],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax2.plot([year, ax1.get_xlim()[-1]], [y_gmst, y_gmst],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:orange")
                set_three_ticks(ax2, df["GMST"], y_gmst)

            xmin, xmax = min(df.index), max(df.index)
            xticks = sorted({xmin, year, xmax})
            for ax in axes:
                ax.set_xticks(xticks)
                ax.set_xticklabels(ax.get_xticks(), rotation=45)
                ax.set_xlim(1850, 2150)
                ax.margins(x=0, y=0)
            plt.tight_layout()
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=( 10, 4), sharex=True)

            # --- Panel 1 (MATLAB figure 2) ---
            ax1 = axes[0]
            ax2 = ax1.twinx()

            ax1.plot(df.index, df["Catm"], color="tab:blue"); ax1.set_ylabel("CO₂ Concentration (ppm)", color="tab:blue"); ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2.plot(df.index, df["RF"], color="tab:green");  ax2.set_ylabel( "Effective Radiative Forcing (W m⁻²)", color="tab:green" ); ax2.tick_params(axis="y", labelcolor="tab:green")
            ax1.set_title("CO₂ concentrations vs Radiative Forcing")
            ax1.set_xlabel("Year")
            ax1.grid(True, which='both', linestyle='--', alpha=0.0)  # Grid on left axis

            # --- Panel 2 (MATLAB figure 3) ---
            axes[1].plot(df.index, df["GMST"], color="tab:orange"); axes[1].set_ylabel("Temperature Anomaly wrt PI (K)", color="tab:orange")
            axes[1].tick_params(axis="y", labelcolor="tab:orange")
            axes[1].set_title("GMST Anomaly")
            axes[1].set_xlabel("Year")
            axes[1].grid(True, which='both', linestyle='--', alpha=0.0)

            if year:
                # Values
                y_catm = df.loc[year, "Catm"]
                y_rf   = df.loc[year, "RF"]
                y_gmst = df.loc[year, "GMST"]

                # --- Panel 1: Catm on left axis ---
                ax1.plot(year, y_catm, marker="o", ms=6, color="tab:blue")
                # vertical line from bottom to point
                ax1.plot([year, year], [ax1.get_ylim()[0], y_catm],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                # horizontal line from left to point
                ax1.plot([ax1.get_xlim()[0], year], [y_catm, y_catm],
                         linestyle=":", alpha=0.6, linewidth=1, color="tab:blue")

                # --- Panel 1: RF on right axis ---
                ax2.plot(year, y_rf, marker="o", ms=6, color="tab:green")
                ax2.plot([year, year], [ax2.get_ylim()[0], y_rf],
                        linestyle=":", alpha=0.6, linewidth=1, color="k")
                ax2.plot([year, ax2.get_xlim()[-1]], [y_rf, y_rf],
                        linestyle=":", alpha=0.6, linewidth=1, color="tab:green")

                # --- Panel 2: GMST ---
                axes[1].plot(year, y_gmst, marker="o", ms=6, color="tab:orange")
                axes[1].plot([year, year], [axes[1].get_ylim()[0], y_gmst],
                            linestyle=":", alpha=0.6, linewidth=1, color="k")
                axes[1].plot([axes[1].get_xlim()[0], year], [y_gmst, y_gmst],
                            linestyle=":", alpha=0.6, linewidth=1, color="tab:orange")

                # X-axis ticks: min, year, max
                xmin, xmax = min(df.index), max(df.index)
                xticks = sorted({xmin, year, xmax})
                for ax in (ax1, ax2, axes[1]):
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(ax.get_xticks(), rotation=45)
                    ax.margins(x=0, y=0)
                    ax.set_xlim(1750, 2100)

                # Y-axis ticks: min, selected, max for each panel
                set_three_ticks(ax1, df["Catm"], y_catm)
                set_three_ticks(ax2, df["RF"], y_rf)
                set_three_ticks(axes[1], df["GMST"], y_gmst)


            plt.tight_layout()
            plt.show()



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

schedule = ContinuousVESchedule(0.01, σmax)


def make_emulator(n_samples=5, n_steps=30):
    emulator_from_pattern = partial(draw_samples_single,
                                    model=nn,
                                    schedule=schedule,
                                    n_samples=n_samples,
                                    n_steps=n_steps,
                                    μ=μ_train,
                                    σ=σ_train)
    # dry run to compile the function
    _ = emulator_from_pattern(pattern=jnp.zeros((96, 192)), key=jr.PRNGKey(0))
    def emulator(ΔT, month, seed):
        pattern = β[month - 1, :, 1] * ΔT + β[month - 1, :, 0]
        pattern = pattern.reshape((96, 192))
        key = jr.PRNGKey(seed)
        samples = emulator_from_pattern(pattern=pattern, key=key)
        return samples
    return emulator


def wrap_as_xarray(samples):
    samples = jnp.concat(samples)
    ds = xr.Dataset(
        {
            var: (('member', 'lat', 'lon'), samples[:, i, :, :])
            for i, var in enumerate(varnames)
        },
        coords={
            'member': jnp.arange(len(samples)) + 1,
            'lat': lat,
            'lon': lon,
        }
    )
    return ds