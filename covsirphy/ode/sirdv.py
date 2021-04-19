#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from covsirphy.ode.mbase import ModelBase


class SIRDV(ModelBase):
    """
    SIR-DV model.

    Args:
        population (int): total population
            kappa (float)
            rho (float)
            sigma (float)
            vacrate (float)
    """
    # Model name
    NAME = "SIR-DV"
    # names of parameters
    PARAMETERS = ["kappa", "rho", "sigma", "vacrate"]
    DAY_PARAMETERS = [
        "1/alpha2 [day]", "1/beta [day]", "1/gamma [day]",
        "Vaccination rate [target population/completion time]"
    ]
    # Variable names in (non-dim, dimensional) ODEs
    VAR_DICT = {
        "x": ModelBase.S,
        "y": ModelBase.CI,
        "z": ModelBase.R,
        "w": ModelBase.F,
        "v": ModelBase.V
    }
    VARIABLES = list(VAR_DICT.values())
    # Weights of variables in parameter estimation error function
    WEIGHTS = np.array([0, 10, 10, 2, 30])
    # Variables that increases monotonically
    VARS_INCLEASE = [ModelBase.R, ModelBase.F]
    # Example set of parameters and initial values
    EXAMPLE = {
        ModelBase.STEP_N: 180,
        ModelBase.N.lower(): 1_000_000,
        ModelBase.PARAM_DICT: {
            "kappa": 0.005, "rho": 0.2, "sigma": 0.075,
            "omega": 0.001,
        },
        ModelBase.Y0_DICT: {
            ModelBase.S: 999_000, ModelBase.CI: 1000, ModelBase.R: 0, ModelBase.F: 0,
            ModelBase.V: 0,
        },
    }

    def __init__(self, population, kappa, rho, sigma,
                 vacrate=None):
        # Total population
        self.population = self._ensure_natural_int(
            population, name="population"
        )
        # Non-dim parameters
        self.kappa = kappa
        self.rho = rho
        self.sigma = sigma
        self.vacrate = vacrate
        self.non_param_dict = {
            "kappa": kappa, "rho": rho, "sigma": sigma, "vacrate": vacrate}

    def __call__(self, t, X):
        """
        Return the list of dS/dt (tau-free) etc.

        Args:
            t (int): time steps
            X (numpy.array): values of th model variables

        Returns:
            (np.array)
        """
        n = self.population
        s, i, *_ = X
        beta_si = self.rho * s * i / n
        dsdt = max(0 - beta_si - self.vacrate, - s)
        dvdt = 0 - dsdt - beta_si
        drdt = self.sigma * i + self.vacrate
        dfdt = self.kappa * i
        didt = 0 - dsdt - drdt - dfdt - dvdt
        return np.array([dsdt, didt, drdt, dfdt, dvdt])

    @classmethod
    def param_range(cls, taufree_df, population, quantiles=(0.1, 0.9)):
        """
        Define the range of parameters (not including tau value).

        Args:
            taufree_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - t (int): time steps (tau-free)
                    - columns with dimensional variables
            population (int): total population

        Returns:
            (dict)
                - key (str): parameter name
                - value (tuple(float, float)): min value and max value
        """
        df = cls._ensure_dataframe(
            taufree_df, name="taufree_df", columns=[cls.TS, *cls.VARIABLES]
        )
        df = df.loc[(df[cls.S] > 0) & (df[cls.CI] > 0)]
        n, t = population, df[cls.TS]
        s, i, r, d = df[cls.S], df[cls.CI], df[cls.R], df[cls.F]
        # kappa = (dD/dt) / I
        kappa_series = d.diff() / t.diff() / i
        # sigma = (dR/dt) / I
        sigma_series = r.diff() / t.diff() / i
        # omega = 0 - (dS/dt + dI/dt + dR/dt + dF/dt)
        vacrate_series = (n - s + i + r + d).diff() / t.diff()
        # Calculate range
        _dict = {
            k: tuple(v.quantile(quantiles).clip(0, 1))
            for (k, v) in zip(
                ["kappa", "sigma", "omega"],
                [kappa_series, sigma_series, vacrate_series]
            )
        }
        _dict["rho"] = (0, 1)
        return _dict

        # _dict = {param: (0, 1) for param in cls.PARAMETERS}
        # if not sigma_series.empty:
        #     _dict["sigma"] = tuple(sigma_series.quantile(
        #         cls.QUANTILE_RANGE).clip(0, 1))
        # if not vacrate_series.empty:
        #     _dict["vacrate"] = tuple(vacrate_series.quantile(
        #         cls.QUANTILE_RANGE).clip(0, 1))
        # return _dict

    @classmethod
    def specialize(cls, data_df, population):
        """
        Specialize the dataset for this model.

        Args:
            data_df (pandas.DataFrame):
                Index
                    reset index
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - any columns
            population (int): total population in the place

        Returns:
            (pandas.DataFrame)
                Index
                    reset index
                Columns
                    - any columns @data_df has
                    - Susceptible (int): 0
                    - Vactinated (int): 0
        """
        df = cls._ensure_dataframe(
            data_df, name="data_df", columns=cls.VALUE_COLUMNS)
        # Calculate dimensional variables
        df[cls.S] = population - df[cls.C]
        df[cls.V] = df[cls.R] - df[cls.CI]
        return df

    @classmethod
    def restore(cls, specialized_df):
        """
        Restore Confirmed/Infected/Recovered/Fatal.
         using a dataframe with the variables of the model.

        Args:
        specialized_df (pandas.DataFrame): dataframe with the variables

            Index
                (object)
            Columns
                - Susceptible (int): the number of susceptible cases
                - Infected (int): the number of currently infected cases
                - Recovered (int): the number of recovered cases
                - Fatal (int): the number of fatal cases
                - Vaccinated (int): the number of vactinated persons
                - any columns

        Returns:
            (pandas.DataFrame)
                Index
                    (object): as-is
                Columns
                    - Confirmed (int): the number of confirmed cases
                    - Infected (int): the number of currently infected cases
                    - Fatal (int): the number of fatal cases
                    - Recovered (int): the number of recovered cases
                    - the other columns @specialzed_df has
        """
        df = specialized_df.copy()
        other_cols = list(set(df.columns) - set(cls.VALUE_COLUMNS))
        df[cls.C] = df[cls.CI] + df[cls.R] + df[cls.F]
        return df.loc[:, [*cls.VALUE_COLUMNS, *other_cols]]

    def calc_r0(self):
        """
        Calculate (basic) reproduction number.

        Returns:
            float
        """
        try:
            rt = self.rho / (self.sigma + self.kappa)
        except ZeroDivisionError:
            return None
        return round(rt, 2)

    def calc_days_dict(self, tau):
        """
        Calculate 1/beta [day] etc.

        Args:
            param tau (int): tau value [min]

        Returns:
            dict[str, int]
        """
        try:
            return {
                "1/alpha2 [day]": int(tau / 24 / 60 / self.kappa),
                "1/beta [day]": int(tau / 24 / 60 / self.rho),
                "1/gamma [day]": int(tau / 24 / 60 / self.sigma),
                "Vaccintion rate [target population/completion time]": float(self.vacrate)
            }
        except ZeroDivisionError:
            return {p: None for p in self.DAY_PARAMETERS}
