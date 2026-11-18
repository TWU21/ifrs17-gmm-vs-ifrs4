"""Tests de coherencia contable del modelo IFRS 17.

Ejecutar desde la raíz del repositorio:  python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ifrs17 import (Basis, Contract, gmm_rollforward, initial_recognition, local_gaap,  # noqa: E402
                    pv_fcf, price, project)

RA = dict(n_sims=4_000)


def mortality() -> pd.Series:
    d = pd.read_csv(ROOT / "data" / "ine_mortality_lx.csv")
    lx = d[(d.tabla_id == "INE_2024") & (d.sexo == "H")].set_index("edad")["lx"]
    return (1 - lx.shift(-1) / lx).dropna()


class TestIFRS17(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = Basis(mortality(), 0.03)
        c = Contract()
        c.premium = price(c, cls.b, margin=0.25)
        cls.c = c
        cls.gmm = gmm_rollforward(c, cls.b, change=(2, 1.2), claims_shock={1: 1.1}, **RA)
        cls.loc = local_gaap(c, cls.b, change=(2, 1.2), claims_shock={1: 1.1})

    def test_equity_identity(self):
        # La variación de (activos - pasivo) de cada año es el resultado del año
        equity = self.gmm["Activos (final)"] - self.gmm["Pasivo IFRS 17 (final)"]
        init = initial_recognition(self.c, self.b, **RA)
        opening = -(init["Flujos de cumplimiento (FCF)"] + init["CSM"])  # = 0 para un grupo rentable
        changes = equity.diff().fillna(equity.iloc[0] - opening)
        np.testing.assert_allclose(changes, self.gmm["Resultado IFRS 17"], rtol=1e-9, atol=1e-6)

    def test_contract_ends_with_no_liability(self):
        self.assertAlmostEqual(self.gmm["CSM (final)"].iloc[-1], 0.0, places=6)
        self.assertAlmostEqual(self.gmm["Pasivo IFRS 17 (final)"].iloc[-1], 0.0, places=4)

    def test_same_total_profit_both_regimes(self):
        # Cambia CUÁNDO se reconoce el beneficio, no CUÁNTO
        self.assertAlmostEqual(self.gmm["Resultado IFRS 17"].sum(),
                               self.loc["Resultado régimen anterior"].sum(), delta=1e-4)

    def test_fcf_roll_forward_consistent(self):
        g = gmm_rollforward(self.c, self.b, **RA)
        l1 = project(self.c, self.b)["inforce"][1]
        self.assertAlmostEqual(g.loc[1, "VA flujos (final)"], float(pv_fcf(project(self.c, self.b, 1, l1), 0.03)), places=4)

    def test_onerous_group_day_one_loss(self):
        c = Contract(premium=price(Contract(), self.b, margin=-0.10))
        init = initial_recognition(c, self.b, **RA)
        self.assertEqual(init["CSM"], 0.0)
        self.assertGreater(init["Pérdida reconocida el día 1"], 0.0)


if __name__ == "__main__":
    unittest.main()
