"""
Modelo general de IFRS 17 para un grupo de seguros temporales de fallecimiento, y comparación
con el régimen anterior (IFRS 4 / PCEA) y con Solvencia II.

Primas y gastos se pagan al inicio de cada año y los siniestros al final. Los flujos de
cumplimiento van con signo de pasivo (salidas menos entradas). Tipo de descuento plano.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Contract:
    age: int = 40
    term: int = 10
    n_policies: int = 1_000
    sum_insured: float = 100_000.0
    premium: float = 0.0              # prima anual de tarifa por póliza
    maintenance: float = 40.0         # gasto de administración por póliza y año
    acquisition: float = 150.0        # gasto de adquisición por póliza (al inicio)
    lapse: float = 0.05               # caídas al final de cada año


@dataclass
class Basis:
    qx: pd.Series                     # q_x de mejor estimación, indexado por edad
    rate: float = 0.03                # tipo de descuento
    mortality_factor: dict = field(default_factory=dict)  # {año desde el que aplica: factor}

    def q(self, age: int, t: int) -> float:
        factor = 1.0
        for start, f in sorted(self.mortality_factor.items()):
            if t >= start:
                factor = f
        return min(1.0, self.qx.loc[age] * factor)


def project(c: Contract, b: Basis, start: int = 0, inforce: float | None = None,
            level: np.ndarray | float = 1.0) -> dict:
    """Flujos esperados desde el año `start` con `inforce` pólizas en vigor.

    `level` es un multiplicador de la mortalidad (escalar o vector de escenarios) que se usa para
    el ajuste por riesgo. Devuelve arrays de forma (n_años,) o (n_escenarios, n_años).
    """
    level = np.atleast_1d(np.asarray(level, float))
    years = np.arange(start, c.term)
    l = np.full(level.shape, c.n_policies if inforce is None else inforce, float)
    out = {k: [] for k in ("inforce", "deaths", "premiums", "expenses", "acquisition", "claims")}
    for t in years:
        q = np.minimum(1.0, b.q(c.age + t, t) * level)
        d = l * q
        out["inforce"].append(l.copy())
        out["deaths"].append(d)
        out["premiums"].append(l * c.premium)
        out["expenses"].append(l * c.maintenance)
        out["acquisition"].append(l * c.acquisition if t == 0 else np.zeros_like(l))
        out["claims"].append(d * c.sum_insured)
        l = (l - d) * (1 - c.lapse)
    res = {k: np.stack(v, axis=-1) for k, v in out.items()}
    res["years"] = years
    if res["inforce"].shape[0] == 1:
        res = {k: (v[0] if k != "years" else v) for k, v in res.items()}
    return res


def price(c: Contract, b: Basis, margin: float) -> float:
    """Prima anual de tarifa: coste esperado (siniestros y gastos) recargado en `margin`."""
    c0 = Contract(**{**c.__dict__, "premium": 1.0})
    p = project(c0, b)
    n = c.term
    v_start, v_end = (1 + b.rate) ** -np.arange(n), (1 + b.rate) ** -np.arange(1, n + 1)
    cost = (p["claims"] * v_end).sum() + ((p["expenses"] + p["acquisition"]) * v_start).sum()
    annuity = (p["premiums"] * v_start).sum()
    return float((1 + margin) * cost / annuity)


def pv_fcf(proj: dict, rate: float) -> np.ndarray:
    """Valor actual (al inicio del primer año proyectado) de salidas menos entradas."""
    n = proj["premiums"].shape[-1]
    v_start = (1 + rate) ** -np.arange(n)
    v_end = (1 + rate) ** -np.arange(1, n + 1)
    start_flows = proj["expenses"] + proj["acquisition"] - proj["premiums"]
    return (start_flows * v_start).sum(-1) + (proj["claims"] * v_end).sum(-1)


def risk_adjustment(c: Contract, b: Basis, start: int = 0, inforce: float | None = None,
                    confidence: float = 0.75, sigma: float = 0.10, n_sims: int = 20_000,
                    seed: int = 17) -> float:
    """Ajuste por riesgo no financiero con la técnica del nivel de confianza.

    La incertidumbre sobre el nivel de mortalidad se representa con un multiplicador lognormal
    de media 1. RA = percentil `confidence` del VA de los flujos menos su media.
    """
    rng = np.random.default_rng(seed)
    level = rng.lognormal(-0.5 * sigma**2, sigma, n_sims)
    pv = pv_fcf(project(c, b, start, inforce, level), b.rate)
    return float(max(0.0, np.quantile(pv, confidence) - pv.mean()))


def initial_recognition(c: Contract, b: Basis, **ra_kwargs) -> pd.Series:
    pv = float(pv_fcf(project(c, b), b.rate))
    ra = risk_adjustment(c, b, **ra_kwargs)
    fcf = pv + ra
    return pd.Series({
        "VA de flujos de salida y entrada": pv,
        "Ajuste por riesgo (RA)": ra,
        "Flujos de cumplimiento (FCF)": fcf,
        "CSM": max(0.0, -fcf),
        "Pérdida reconocida el día 1": max(0.0, fcf),
    })


def gmm_rollforward(c: Contract, b: Basis, change: tuple[int, float] | None = None,
                    claims_shock: dict | None = None, **ra_kwargs) -> pd.DataFrame:
    """Evolución anual de un grupo rentable con el modelo general.

    change       = (año t, factor): al final del año t, la mortalidad esperada para los años
                   siguientes pasa a multiplicarse por `factor` (cambio de hipótesis, servicio futuro).
    claims_shock = {año t: factor}: siniestralidad real distinta de la esperada (servicio pasado).
    """
    claims_shock = claims_shock or {}
    i = b.rate
    base = project(c, b)
    init = initial_recognition(c, b, **ra_kwargs)
    if init["CSM"] == 0:
        raise ValueError("El grupo es oneroso: esta función solo modela grupos rentables.")

    cu = base["inforce"] * c.sum_insured               # unidades de cobertura previstas al inicio
    acq_total = base["acquisition"].sum()
    pv_open, ra_open, csm_open, assets = init["VA de flujos de salida y entrada"], init["Ajuste por riesgo (RA)"], init["CSM"], 0.0
    basis, l = b, float(c.n_policies)
    rows = []
    for t in range(c.term):
        p = project(c, basis, t, l)
        prem, exp_, acq = p["premiums"][0], p["expenses"][0], p["acquisition"][0]
        exp_claims = p["claims"][0]
        act_claims = exp_claims * claims_shock.get(t, 1.0)
        l_next = (l - p["deaths"][0]) * (1 - c.lapse)
        last = t == c.term - 1

        # Flujos de cumplimiento
        pv_after = pv_open + prem - exp_ - acq
        fin_fcf = i * pv_after
        pv_end = pv_after * (1 + i) - exp_claims
        ra_end = 0.0 if last else risk_adjustment(c, basis, t + 1, l_next, **ra_kwargs)
        ra_release = ra_open - ra_end

        # CSM: intereses al tipo bloqueado, cambios de servicio futuro y liberación
        csm_interest = i * csm_open
        csm = csm_open + csm_interest
        future_change = 0.0
        if change is not None and t == change[0] and not last:
            new_basis = Basis(basis.qx, basis.rate, {**basis.mortality_factor, t + 1: change[1]})
            pv_new = float(pv_fcf(project(c, new_basis, t + 1, l_next), i))
            ra_new = risk_adjustment(c, new_basis, t + 1, l_next, **ra_kwargs)
            future_change = (pv_new - pv_end) + (ra_new - ra_end)
            pv_end, ra_end, basis = pv_new, ra_new, new_basis
            if future_change > csm:
                raise ValueError("El cambio vuelve oneroso el grupo; fuera del alcance de este ejemplo.")
            csm -= future_change
        # Unidades de cobertura del año frente a las del año y las futuras esperadas
        future_cu = np.atleast_1d(project(c, basis, t, l)["inforce"]) * c.sum_insured
        release = csm * future_cu[0] / future_cu.sum()
        csm_end = csm - release
        l = l_next

        # Cuenta de resultados
        acq_recovery = acq_total * cu[t] / cu.sum()
        revenue = exp_claims + exp_ + ra_release + release + acq_recovery
        service_expenses = act_claims + exp_ + acq_recovery
        inv_base = assets + prem - exp_ - acq
        inv_income = i * inv_base
        fin_expense = fin_fcf + csm_interest
        assets = inv_base * (1 + i) - act_claims

        rows.append({
            "Año": t + 1,
            "Primas cobradas": prem,
            "Ingresos de seguros": revenue,
            "Gastos del servicio de seguros": service_expenses,
            "Resultado del servicio de seguros": revenue - service_expenses,
            "Ingresos financieros": inv_income,
            "Gastos financieros de seguros": fin_expense,
            "Resultado IFRS 17": revenue - service_expenses + inv_income - fin_expense,
            "Intereses del CSM": csm_interest,
            "Liberación del CSM": release,
            "Liberación del RA": ra_release,
            "Ajuste del CSM por servicio futuro": -future_change,
            "VA flujos (final)": pv_end,
            "RA (final)": ra_end,
            "CSM (final)": csm_end,
            "Pasivo IFRS 17 (final)": pv_end + ra_end + csm_end,
            "Activos (final)": assets,
        })
        pv_open, ra_open, csm_open = pv_end, ra_end, csm_end
    return pd.DataFrame(rows).set_index("Año")


def local_gaap(c: Contract, b: Basis, technical_rate: float = 0.02, technical_loading: float = 0.10,
               change: tuple[int, float] | None = None, claims_shock: dict | None = None) -> pd.DataFrame:
    """Régimen anterior simplificado (IFRS 4 con criterio local tipo PCEA).

    - Ingreso = primas emitidas; los gastos de adquisición van a resultados cuando se pagan.
    - Provisión matemática prospectiva con bases técnicas prudentes fijadas al emitir
      (mortalidad +10 %, interés técnico del 2 %, sin caídas) que no se revisan.
    - Los flujos reales siguen la misma experiencia que en el modelo IFRS 17.
    """
    claims_shock = claims_shock or {}
    i = b.rate
    n = c.term
    q_tech = np.array([min(1.0, b.qx.loc[c.age + t] * (1 + technical_loading)) for t in range(n)])
    v = 1 / (1 + technical_rate)

    def reserve_per_policy(t):  # al inicio del año t, antes de cobrar la prima
        surv, ben, ann = 1.0, 0.0, 0.0
        for k in range(t, n):
            ann += surv * v ** (k - t)
            ben += surv * q_tech[k] * v ** (k - t + 1)
            surv *= 1 - q_tech[k]
        return ben * c.sum_insured, ann

    ben0, ann0 = reserve_per_policy(0)
    net_premium = ben0 / ann0

    def reserve_end(t):  # al final del año t por póliza superviviente (= inicio de t+1)
        if t + 1 >= n:
            return 0.0
        ben, ann = reserve_per_policy(t + 1)
        return ben - net_premium * ann

    # La experiencia real es la misma que en el modelo IFRS 17 (incluido el cambio de mortalidad)
    actual = Basis(b.qx, b.rate, {change[0] + 1: change[1]}) if change is not None else b
    l = float(c.n_policies)
    assets, res_open, rows = 0.0, 0.0, []
    for t in range(n):
        p = project(c, actual, t, l)
        prem, exp_, acq = p["premiums"][0], p["expenses"][0], p["acquisition"][0]
        act_claims = p["claims"][0] * claims_shock.get(t, 1.0)
        survivors_end = (l - p["deaths"][0]) * (1 - c.lapse)
        l = survivors_end
        res_end = reserve_end(t) * survivors_end
        inv_base = assets + prem - exp_ - acq
        inv_income = i * inv_base
        assets = inv_base * (1 + i) - act_claims
        profit = prem - exp_ - acq - act_claims - (res_end - res_open) + inv_income
        rows.append({
            "Año": t + 1,
            "Primas emitidas (ingreso)": prem,
            "Siniestros": act_claims,
            "Gastos (incl. adquisición)": exp_ + acq,
            "Variación de la provisión matemática": res_end - res_open,
            "Ingresos financieros": inv_income,
            "Resultado régimen anterior": profit,
            "Provisión matemática (final)": res_end,
            "Activos (final)": assets,
        })
        res_open = res_end
    return pd.DataFrame(rows).set_index("Año")


def solvency_ii_inception(c: Contract, b: Basis, mortality_shock: float = 0.15,
                          coc: float = 0.06) -> pd.Series:
    """BEL y margen de riesgo (coste de capital) al inicio.

    SCR de mortalidad = aumento de la BEL ante un +15 % permanente en q_x (fórmula estándar),
    proyectado año a año sobre la cartera esperada. RM = 6 % × Σ SCR_t descontados.
    """
    i = b.rate
    base = project(c, b)
    shocked = Basis(b.qx, b.rate, {0: 1 + mortality_shock})
    scr = []
    for t in range(c.term):
        bel_t = pv_fcf(project(c, b, t, base["inforce"][t]), i)
        bel_s = pv_fcf(project(c, shocked, t, base["inforce"][t]), i)
        scr.append(max(0.0, float(bel_s - bel_t)))
    scr = np.array(scr)
    rm = coc * float(np.sum(scr * (1 + i) ** -np.arange(1, c.term + 1)))
    bel = float(pv_fcf(base, i))
    return pd.Series({
        "Mejor estimación (BEL)": bel,
        "Margen de riesgo (RM)": rm,
        "Provisiones técnicas (BEL + RM)": bel + rm,
        "SCR de mortalidad en t = 0": scr[0],
    })
