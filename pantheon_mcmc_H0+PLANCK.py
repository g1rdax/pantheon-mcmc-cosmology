#!/usr/bin/env python3
"""Оцінювання космологічних параметрів за Pantheon+SH0ES.

Скрипт читає Pantheon+SH0ES.dat та повну матрицю коваріації
Pantheon+SH0ES_STAT+SYS.cov, оцінює параметри моделей плоскої ΛCDM або
CPL w0waCDM за допомогою emcee та створює діаграму Габбла і кутові контури.

У функції правдоподібності використовується m_b_corr разом із CEPH_DIST,
IS_CALIBRATOR та повною матрицею коваріації. Для SN у цефеїдних галактиках
відстань порівнюється з незалежною відстанню цефеїд; для решти об'єктів
використовується космологічна модель. Абсолютна зоряна величина M_B
семплюється як параметр завади, тож калібратори розривають виродження M_B–H0.
Рядки з zHD <= 0.01, що не є калібраторами, виключаються з космологічного
порівняння згідно з офіційною схемою аналізу.

Стовпці m_b_corr_err_DIAG та MU_SH0ES_ERR_DIAG є діагональними похибками
для візуалізації; вони не підміняють повну коваріацію у функції правдоподібності.

Швидкий режим MCMC використовує векторизовану квадратуру Гаусса—Лежандра,
а незалежна перевірка виконується адаптивним scipy.integrate.quad. Для
паралельної оцінки правдоподіб'я застосовується spawn-сумісний пул процесів.
Модель ΛCDM фіксує w0=-1 та wa=0 і не семплює ці параметри.

Залежності:
    numpy, pandas, scipy, emcee, corner, matplotlib

Приклад запуску з кореня розпакованого DataRelease-main.zip:
    python outputs/pantheon_mcmc.py --model lcdm
    python outputs/pantheon_mcmc.py --model w0wacdm

Якщо файли релізу лежать в іншому місці, передайте --data та --covariance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import matplotlib

    # Agg працює без графічного дисплея на серверах і кластерах.
    matplotlib.use("Agg", force=True)
    import corner
    import emcee
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.integrate import IntegrationWarning, quad
    from scipy.linalg import cho_factor, cho_solve, eigh
except ImportError as exc:  # Повідомлення допомагає встановити відсутню залежність.
    raise SystemExit(
        "Не знайдено потрібну Python-бібліотеку. Встановіть залежності командою: "
        "python -m pip install numpy pandas scipy emcee corner matplotlib"
    ) from exc


LOGGER = logging.getLogger("pantheon_mcmc")
C_KM_S = 299_792.458
FULL_PARAMETER_NAMES = ("H0", "Omega_m", "w0", "wa", "M_B")
SAMPLED_PARAMETER_NAMES = {
    "lcdm": ("H0", "Omega_m", "M_B"),
    "w0wacdm": FULL_PARAMETER_NAMES,
}
CONTOUR_PARAMETER_LABELS = {
    "H0": r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    "Omega_m": r"$\Omega_m$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}
COSMOLOGICAL_CONTOUR_PARAMETERS = ("H0", "Omega_m", "w0", "wa")
REQUIRED_COLUMNS = (
    "CID",
    "IDSURVEY",
    "zHD",
    "zHEL",
    "m_b_corr",
    "m_b_corr_err_DIAG",
    "CEPH_DIST",
    "IS_CALIBRATOR",
)
RELEASE_SUBDIRECTORY = Path("Pantheon+_Data") / "4_DISTANCES_AND_COVAR"
GL_NODES_64, GL_WEIGHTS_64 = np.polynomial.legendre.leggauss(64)
_WORKER_LIKELIHOOD: GaussianLikelihood | None = None


@dataclass(frozen=True)
class PantheonData:
    """Масив даних зі збереженим порядком рядків офіційного релізу."""

    table: pd.DataFrame
    names: np.ndarray
    survey_ids: np.ndarray
    redshift: np.ndarray
    heliocentric_redshift: np.ndarray
    apparent_magnitude: np.ndarray
    apparent_magnitude_error: np.ndarray
    cepheid_distance: np.ndarray
    is_calibrator: np.ndarray
    use_for_cosmology: np.ndarray


@dataclass(frozen=True)
class CovarianceMetric:
    """Підготовлена метрика коваріації для гаусового правдоподіб'я."""

    precision: np.ndarray
    log_determinant: float
    rank: int
    method: str
    null_basis: np.ndarray | None = None


@dataclass(frozen=True)
class MCMCResult:
    """Ланцюг продукційної фази та діагностики семплера."""

    samples: np.ndarray
    maximum_a_posteriori: np.ndarray
    acceptance_fraction: np.ndarray
    autocorrelation_time: np.ndarray | None


def setup_logging(verbose: bool = False) -> None:
    """Налаштовує короткі повідомлення про перебіг аналізу українською."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def resolve_release_file(filename: str, explicit_path: Path | None) -> Path:
    """Знаходить файл релізу за заданим шляхом або типовими каталогами."""
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Файл не знайдено: {candidate}")

    relative = RELEASE_SUBDIRECTORY / filename
    roots = (
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
    )
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root / "DataRelease-main" / relative,
                root / relative,
                root / "data" / filename,
            )
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    formatted = "\n".join(f"  - {item}" for item in candidates)
    raise FileNotFoundError(
        f"Не вдалося знайти {filename}. Розпакуйте DataRelease-main.zip або "
        f"передайте явний шлях через відповідний аргумент командного рядка.\n"
        f"Перевірені шляхи:\n{formatted}"
    )


def load_pantheon_data(path: Path) -> PantheonData:
    """Зчитує таблицю, не змінюючи її порядок, потрібний для коваріації.

    Аргументи:
        path: шлях до Pantheon+SH0ES.dat.

    Повертає:
        Об'єкт із назвами SN, zHD, m_b_corr, діагональними похибками та
        параметрами калібраторів та zHEL.

    Піднімає:
        FileNotFoundError: якщо таблиці немає.
        ValueError: якщо бракує колонок або дані некоректні.
    """
    try:
        table = pd.read_csv(path, sep=r"\s+", comment="#")
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise RuntimeError(f"Не вдалося прочитати таблицю {path}: {exc}") from exc

    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(
            "У таблиці Pantheon+ відсутні обов'язкові колонки: "
            + ", ".join(missing)
        )
    if table.empty:
        raise ValueError("Таблиця Pantheon+ не містить жодного рядка даних.")

    numeric_columns = (
        "IDSURVEY",
        "zHD",
        "zHEL",
        "m_b_corr",
        "m_b_corr_err_DIAG",
        "CEPH_DIST",
        "IS_CALIBRATOR",
    )
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")

    numeric_values = table.loc[:, list(numeric_columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("У числових колонках таблиці є NaN або нескінченні значення.")
    if np.any(table["zHD"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("Усі значення zHD мають бути додатними для логарифмічної осі.")
    if np.any(table["zHEL"].to_numpy(dtype=float) < 0.0):
        raise ValueError("Значення zHEL мають бути невід'ємними.")
    for column in ("m_b_corr_err_DIAG", "MU_SH0ES_ERR_DIAG"):
        if column in table.columns and np.any(
            table[column].to_numpy(dtype=float) <= 0.0
        ):
            raise ValueError(f"Похибки у колонці {column} мають бути додатними.")

    calibrator_flags = table["IS_CALIBRATOR"].to_numpy(dtype=float)
    if not np.all(np.isin(calibrator_flags, (0.0, 1.0))):
        raise ValueError("IS_CALIBRATOR має містити лише значення 0 або 1.")
    is_calibrator = calibrator_flags.astype(bool)
    cepheid_distance = table["CEPH_DIST"].to_numpy(dtype=float)
    if not np.any(is_calibrator):
        raise ValueError("У таблиці не знайдено цефеїдних калібраторів.")
    if np.any(cepheid_distance[is_calibrator] <= 0.0):
        raise ValueError("CEPH_DIST має бути додатною для кожного калібратора.")
    use_for_cosmology = (table["zHD"].to_numpy(dtype=float) > 0.01) | is_calibrator
    if not np.any(use_for_cosmology & ~is_calibrator):
        raise ValueError("Не знайдено Hubble-flow наднових із zHD > 0.01.")

    # CID може повторюватися, бо таблиця містить 1701 криву блиску, а не
    # 1701 різну наднову; повтори зберігаємо для узгодження з матрицею.
    names = table["CID"].astype(str).to_numpy()
    return PantheonData(
        table=table,
        names=names,
        survey_ids=table["IDSURVEY"].to_numpy(dtype=int),
        redshift=table["zHD"].to_numpy(dtype=float),
        heliocentric_redshift=table["zHEL"].to_numpy(dtype=float),
        apparent_magnitude=table["m_b_corr"].to_numpy(dtype=float),
        apparent_magnitude_error=table["m_b_corr_err_DIAG"].to_numpy(dtype=float),
        cepheid_distance=cepheid_distance,
        is_calibrator=is_calibrator,
        use_for_cosmology=use_for_cosmology,
    )


def load_covariance_matrix(path: Path, expected_size: int | None = None) -> np.ndarray:
    """Завантажує повну матрицю з формату релізу: N, а потім N² чисел.

    Перший рядок .cov задає розмір матриці. Подальші значення читаються у
    послідовному порядку та формують симетричну матрицю N × N.
    """
    try:
        with path.open("r", encoding="ascii") as stream:
            first_line = stream.readline().strip()
        size = int(first_line)
        if size <= 0:
            raise ValueError("Розмір коваріаційної матриці має бути додатним.")
        if expected_size is not None and size != expected_size:
            raise ValueError(
                f"Розмір коваріації N={size} не збігається з кількістю "
                f"рядків таблиці ({expected_size}). Перевірте, що файли "
                "з одного релізу."
            )
        flat_values = np.loadtxt(path, dtype=np.float64, skiprows=1)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"Не вдалося прочитати матрицю коваріації {path}: {exc}"
        ) from exc

    if flat_values.size != size * size:
        raise ValueError(
            f"У {path} очікувалося {size * size:,} елементів після заголовка, "
            f"отримано {flat_values.size:,}."
        )
    covariance = np.asarray(flat_values, dtype=np.float64).reshape(size, size)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Матриця коваріації містить NaN або нескінченні значення.")

    # Текстовий реліз округлено десятковим записом, тому дозволяємо лише
    # мізерну відносну асиметрію та прибираємо її середнім C і Cᵀ.
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    relative_asymmetry = float(np.max(np.abs(covariance - covariance.T)) / scale)
    if relative_asymmetry > 5.0e-6:
        raise ValueError(
            "Матриця коваріації помітно несиметрична: відносна асиметрія "
            f"{relative_asymmetry:.3e}. Перевірте файл або порядок його чисел."
        )
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) <= 0.0):
        raise ValueError("Діагональні елементи коваріації мають бути додатними.")
    return covariance


def prepare_covariance(
    covariance: np.ndarray,
    allow_pseudoinverse: bool = False,
    rcond: float = 1.0e-12,
) -> CovarianceMetric:
    """Обчислює матрицю точності та log|C| із перевіркою позитивної визначеності.

    Основний шлях використовує розклад Холєцького та розв'язання системи
    C X = I; це чисельно надійніше, ніж безпосередньо обертати C. Якщо
    розклад Холєцького не вдається, псевдообернення дозволяється лише за
    явним прапорцем командного рядка.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Матриця коваріації має бути квадратною.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Матриця коваріації містить NaN або нескінченні значення.")
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    relative_asymmetry = float(np.max(np.abs(covariance - covariance.T)) / scale)
    if relative_asymmetry > 5.0e-6:
        raise ValueError(
            "Матриця коваріації помітно несиметрична: відносна асиметрія "
            f"{relative_asymmetry:.3e}."
        )
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) <= 0.0):
        raise ValueError("Діагональні елементи коваріації мають бути додатними.")
    try:
        factor, lower = cho_factor(covariance, lower=True, check_finite=False)
        identity = np.eye(covariance.shape[0], dtype=np.float64)
        precision = cho_solve((factor, lower), identity, check_finite=False)
        precision = 0.5 * (precision + precision.T)
        log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor))))
        if not np.isfinite(log_determinant) or not np.all(np.isfinite(precision)):
            raise np.linalg.LinAlgError("Невизначений результат розкладу Холєцького.")
        return CovarianceMetric(
            precision=precision,
            log_determinant=log_determinant,
            rank=covariance.shape[0],
            method="Cholesky solve",
            null_basis=None,
        )
    except np.linalg.LinAlgError as exc:
        if not allow_pseudoinverse:
            raise ValueError(
                "Коваріація не є додатно визначеною. Перевірте, що це повна "
                "матриця STAT+SYS релізу. Псевдообернення можна дозволити "
                "аргументом --allow-pseudoinverse, якщо виродженість очікувана."
            ) from exc

    eigenvalues, eigenvectors = eigh(covariance, check_finite=False)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    cutoff = rcond * spectral_scale
    if float(np.min(eigenvalues)) < -cutoff:
        raise ValueError(
            "Матриця коваріації має суттєво від'ємні власні значення; "
            "псевдообернення не може виправити таку матрицю."
        )
    retained = eigenvalues > cutoff
    rank = int(np.count_nonzero(retained))
    if rank == 0:
        raise ValueError("Псевдообернена матриця має нульовий ранг.")
    retained_values = eigenvalues[retained]
    retained_vectors = eigenvectors[:, retained]
    precision = (retained_vectors / retained_values) @ retained_vectors.T
    precision = 0.5 * (precision + precision.T)
    log_pseudo_determinant = float(np.sum(np.log(retained_values)))
    return CovarianceMetric(
        precision=precision,
        log_determinant=log_pseudo_determinant,
        rank=rank,
        method=f"eigenvalue pseudoinverse (rank {rank}/{covariance.shape[0]})",
        null_basis=eigenvectors[:, ~retained],
    )


def expansion_rate_e(
    redshift: np.ndarray | float,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
) -> np.ndarray:
    """Повертає E(z) для плоскої CPL-моделі w0waCDM.

    Для ΛCDM залишаються значення за замовчуванням ``w0=-1`` і ``wa=0``.
    """
    z = np.asarray(redshift, dtype=np.float64)
    if not (0.0 < omega_m < 1.0):
        raise ValueError("Для цієї моделі потрібно 0 < Omega_m < 1.")
    if np.any(z < 0.0) or not np.all(np.isfinite(z)):
        raise ValueError("Червоні зміщення мають бути скінченними та невід'ємними.")
    if not np.all(np.isfinite((w0, wa))):
        raise ValueError("Параметри w0 та wa мають бути скінченними.")
    one_plus_z = 1.0 + z
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            dark_energy = (1.0 - omega_m) * np.power(
                one_plus_z, 3.0 * (1.0 + w0 + wa)
            ) * np.exp(-3.0 * wa * z / one_plus_z)
            value = omega_m * np.power(one_plus_z, 3.0) + dark_energy
        except FloatingPointError as exc:
            raise ValueError("Не вдалося стабільно обчислити E(z).") from exc
    if np.any(value <= 0.0) or not np.all(np.isfinite(value)):
        raise ValueError("Розширення E(z) не є дійсним і додатним.")
    return np.sqrt(value)


def luminosity_distance_mpc(
    redshift: np.ndarray | float,
    h0: float,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
    redshift_hel: np.ndarray | float | None = None,
    epsabs: float = 1.0e-10,
    epsrel: float = 1.0e-10,
) -> np.ndarray | float:
    """Обчислює D_L у Мпк адаптивним інтегруванням scipy.integrate.quad.

    Метод виконує адаптивний інтеграл для кожного елемента z і призначений
    для перевірки та побудови гладкої кривої. Для кожної точки MCMC нижче
    використано еквівалентну векторизовану квадратуру високого порядку.
    """
    z_input = np.asarray(redshift, dtype=np.float64)
    z_hel_input = z_input if redshift_hel is None else np.asarray(
        redshift_hel, dtype=np.float64
    )
    z, z_hel = np.broadcast_arrays(z_input, z_hel_input)
    if h0 <= 0.0 or not np.isfinite(h0):
        raise ValueError("H0 має бути скінченним додатним числом.")
    if (
        np.any(z < 0.0)
        or np.any(z_hel < 0.0)
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(z_hel))
    ):
        raise ValueError("Червоні зміщення мають бути скінченними та невід'ємними.")
    if not (0.0 < omega_m < 1.0):
        raise ValueError("Для цієї моделі потрібно 0 < Omega_m < 1.")

    flat_z = z.reshape(-1)
    integrals = np.empty_like(flat_z)
    for index, z_value in enumerate(flat_z):
        if z_value == 0.0:
            integrals[index] = 0.0
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                integral, _ = quad(
                    lambda z_prime: 1.0 / float(
                        expansion_rate_e(z_prime, omega_m, w0=w0, wa=wa)
                    ),
                    0.0,
                    float(z_value),
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=200,
                )
        except (IntegrationWarning, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Адаптивне інтегрування не вдалося для z={z_value:.8g}: {exc}"
            ) from exc
        integrals[index] = integral

    flat_z_hel = z_hel.reshape(-1)
    # Інтеграл задає поперечну комовну відстань; множник zHEL переводить її
    # у luminosity distance у потрібній системі відліку.
    luminosity_distance = (C_KM_S / h0) * (1.0 + flat_z_hel) * integrals
    luminosity_distance = luminosity_distance.reshape(z.shape)
    if z.ndim == 0:
        return float(luminosity_distance)
    return luminosity_distance


def luminosity_distance_gauss_mpc(
    redshift: np.ndarray | float,
    h0: float,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
    redshift_hel: np.ndarray | float | None = None,
    order: int = 64,
) -> np.ndarray | float:
    """Швидко обчислює D_L для всього вектора червоних зміщень.

    Перетворює інтеграл на відрізок [-1, 1] та оцінює його квадратурою
    Гаусса—Лежандра. Функція підходить для багаторазових викликів MCMC.
    """
    z_input = np.asarray(redshift, dtype=np.float64)
    z_hel_input = z_input if redshift_hel is None else np.asarray(
        redshift_hel, dtype=np.float64
    )
    z, z_hel = np.broadcast_arrays(z_input, z_hel_input)
    if h0 <= 0.0 or not np.isfinite(h0):
        raise ValueError("H0 має бути скінченним додатним числом.")
    if (
        np.any(z < 0.0)
        or np.any(z_hel < 0.0)
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(z_hel))
    ):
        raise ValueError("Червоні зміщення мають бути скінченними та невід'ємними.")
    if not (0.0 < omega_m < 1.0):
        raise ValueError("Для цієї моделі потрібно 0 < Omega_m < 1.")
    if order < 16:
        raise ValueError("Порядок квадратури має бути не меншим за 16.")

    if order == 64:
        nodes, weights = GL_NODES_64, GL_WEIGHTS_64
    else:
        nodes, weights = np.polynomial.legendre.leggauss(order)
    flat_z = z.reshape(-1)
    flat_z_hel = z_hel.reshape(-1)
    evaluation_points = 0.5 * flat_z[:, None] * (nodes[None, :] + 1.0)
    integrand = 1.0 / expansion_rate_e(
        evaluation_points, omega_m, w0=w0, wa=wa
    )
    integrals = 0.5 * flat_z * np.sum(integrand * weights[None, :], axis=1)
    luminosity_distance = (
        (C_KM_S / h0)
        * (1.0 + flat_z_hel)
        * integrals
    )
    luminosity_distance = luminosity_distance.reshape(z.shape)
    if z.ndim == 0:
        return float(luminosity_distance)
    return luminosity_distance


def distance_modulus(
    redshift: np.ndarray | float,
    h0: float,
    omega_m: float,
    w0: float = -1.0,
    wa: float = 0.0,
    redshift_hel: np.ndarray | float | None = None,
    integration: str = "gauss",
    order: int = 64,
) -> np.ndarray | float:
    """Обчислює μ=5 log10(D_L/Мпк)+25 обраним методом інтегрування."""
    if integration == "quad":
        distance = luminosity_distance_mpc(
            redshift, h0, omega_m, w0=w0, wa=wa, redshift_hel=redshift_hel
        )
    elif integration == "gauss":
        distance = luminosity_distance_gauss_mpc(
            redshift,
            h0,
            omega_m,
            w0=w0,
            wa=wa,
            redshift_hel=redshift_hel,
            order=order,
        )
    else:
        raise ValueError("integration має бути 'quad' або 'gauss'.")
    distance_array = np.asarray(distance, dtype=np.float64)
    if np.any(distance_array <= 0.0):
        raise ValueError("Для модуля відстані потрібно D_L > 0.")
    result = 5.0 * np.log10(distance_array) + 25.0
    if result.ndim == 0:
        return float(result)
    return result


def validate_quadrature(
    order: int, maximum_redshift: float, model: str = "lcdm"
) -> float:
    """Порівнює швидку квадратуру з адаптивним quad на контрольній сітці."""
    redshifts = np.linspace(maximum_redshift / 100.0, maximum_redshift, 7)
    maximum_difference = 0.0
    if model not in SAMPLED_PARAMETER_NAMES:
        raise ValueError(f"Невідома модель: {model}")
    dark_energy_cases = ((-1.0, 0.0),) if model == "lcdm" else (
        (-1.5, -2.5),
        (-1.0, 0.0),
        (0.5, 2.5),
    )
    for omega_m in (0.2, 0.33, 0.7):
        for w0, wa in dark_energy_cases:
            adaptive = distance_modulus(
                redshifts, 70.0, omega_m, w0=w0, wa=wa, integration="quad"
            )
            fast = distance_modulus(
                redshifts,
                70.0,
                omega_m,
                w0=w0,
                wa=wa,
                integration="gauss",
                order=order,
            )
            maximum_difference = max(
                maximum_difference, float(np.max(np.abs(adaptive - fast)))
            )
    if maximum_difference > 1.0e-8:
        raise RuntimeError(
            "Швидка квадратура не пройшла перевірку відносно quad: "
            f"максимальна різниця {maximum_difference:.3e} mag."
        )
    return maximum_difference


class GaussianLikelihood:
    """Гаусова правдоподібність з повною матрицею коваріації."""

    def __init__(
        self,
        data: PantheonData,
        metric: CovarianceMetric,
        integration_order: int = 64,
        model: str = "lcdm",
    ) -> None:
        if model not in SAMPLED_PARAMETER_NAMES:
            raise ValueError(f"Невідома космологічна модель: {model}")
        mask = data.use_for_cosmology
        self.redshift = data.redshift[mask]
        self.redshift_hel = data.heliocentric_redshift[mask]
        self.apparent_magnitude = data.apparent_magnitude[mask]
        self.cepheid_distance = data.cepheid_distance[mask]
        self.is_calibrator = data.is_calibrator[mask]
        self.precision = metric.precision
        self.log_determinant = metric.log_determinant
        self.rank = metric.rank
        self.null_basis = metric.null_basis
        self.integration_order = integration_order
        self.model = model
        self.parameter_names = SAMPLED_PARAMETER_NAMES[model]
        if self.precision.shape != (self.redshift.size, self.redshift.size):
            raise ValueError("Розмір матриці точності не узгоджується з таблицею.")
        self.normalization = self.log_determinant + self.rank * math.log(2.0 * math.pi)

        # Забороняємо випадкові зміни спільних знімків даних у головному
        # процесі або в дочірніх процесах пулу.
        for array in (
            self.redshift,
            self.redshift_hel,
            self.apparent_magnitude,
            self.cepheid_distance,
            self.is_calibrator,
            self.precision,
        ):
            array.setflags(write=False)
        if self.null_basis is not None:
            self.null_basis.setflags(write=False)

    def residuals(self, parameters: Sequence[float]) -> np.ndarray | None:
        """Формує вектор залишків для заданих активних параметрів моделі."""
        try:
            h0, omega_m, w0, wa, absolute_magnitude = expand_parameters(
                parameters, self.model
            )
            model_distance = np.asarray(
                distance_modulus(
                    self.redshift,
                    h0,
                    omega_m,
                    w0=w0,
                    wa=wa,
                    redshift_hel=self.redshift_hel,
                    integration="gauss",
                    order=self.integration_order,
                ),
                dtype=np.float64,
            )
        except (ValueError, FloatingPointError, OverflowError, IndexError):
            return None
        # Калібратори порівнюються з незалежними відстанями цефеїд;
        # їхнє червоне зміщення не є космологічним предиктором.
        model_distance[self.is_calibrator] = self.cepheid_distance[
            self.is_calibrator
        ]
        residual = self.apparent_magnitude - (model_distance + absolute_magnitude)
        if not np.all(np.isfinite(residual)):
            return None
        if self.null_basis is not None and self.null_basis.size:
            unsupported = self.null_basis.T @ residual
            tolerance = 1.0e-8 * max(1.0, float(np.linalg.norm(residual)))
            if float(np.linalg.norm(unsupported)) > tolerance:
                return None
        return residual

    def chi_squared(self, parameters: Sequence[float]) -> float:
        """Повертає χ² для моделі з повною матрицею точності."""
        residual = self.residuals(parameters)
        if residual is None:
            return float("inf")
        quadratic = float(residual @ (self.precision @ residual))
        if not np.isfinite(quadratic):
            return float("inf")
        if quadratic < 0.0:
            # Лише мікроскопічне від'ємне значення можна пояснити округленням.
            if quadratic > -1.0e-9:
                return 0.0
            return float("inf")
        return quadratic

    def __call__(self, parameters: Sequence[float]) -> float:
        """Повертає повний логарифм багатовимірної нормальної густини."""
        chi_squared = self.chi_squared(parameters)
        if not np.isfinite(chi_squared):
            return -np.inf
        return -0.5 * (chi_squared + self.normalization)


def expand_parameters(
    parameters: Sequence[float], model: str
) -> tuple[float, float, float, float, float]:
    """Розгортає активні параметри у єдиний фізичний порядок H0, Ωm, w0, wa, M_B.

    У базовій ΛCDM-фазі sampler має лише три координати, а ``w0=-1`` і
    ``wa=0`` фіксуються точно, а не наближаються вузькими апріорними межами.
    """
    if model == "lcdm":
        if len(parameters) != 3:
            raise ValueError("LCDM очікує параметри (H0, Omega_m, M_B).")
        h0, omega_m, absolute_magnitude = map(float, parameters)
        return h0, omega_m, -1.0, 0.0, absolute_magnitude
    if model == "w0wacdm":
        if len(parameters) != 5:
            raise ValueError("w0waCDM очікує п'ять параметрів.")
        h0, omega_m, w0, wa, absolute_magnitude = map(float, parameters)
        return h0, omega_m, w0, wa, absolute_magnitude
    raise ValueError(f"Невідома космологічна модель: {model}")


def log_prior(parameters: Sequence[float], model: str = "lcdm") -> float:
    """Рівномірні апріорні розподіли у заданих фізичних межах."""
    if model not in SAMPLED_PARAMETER_NAMES:
        return -np.inf
    if len(parameters) != len(SAMPLED_PARAMETER_NAMES[model]):
        return -np.inf
    try:
        h0, omega_m, w0, wa, absolute_magnitude = expand_parameters(
            parameters, model
        )
    except (TypeError, ValueError):
        return -np.inf
    if not all(
        np.isfinite(value)
        for value in (h0, omega_m, w0, wa, absolute_magnitude)
    ):
        return -np.inf
    common_bounds = (
        50.0 < h0 < 90.0
        and 0.0 < omega_m < 1.0
        and -20.5 < absolute_magnitude < -18.0
    )
    if model == "lcdm":
        return 0.0 if common_bounds else -np.inf
    if common_bounds and -3.0 < w0 < 1.0 and -3.0 < wa < 3.0:
        return 0.0
    return -np.inf


def log_probability(
    parameters: Sequence[float], likelihood: GaussianLikelihood
) -> float:
    """Складає логарифми апріорного розподілу та правдоподібності."""
    prior = log_prior(parameters, likelihood.model)
    if not np.isfinite(prior):
        return -np.inf
    return prior + likelihood(parameters)


def initialize_likelihood_worker(likelihood: GaussianLikelihood) -> None:
    """Ініціалізує незмінне правдоподіб'я один раз у кожному процесі."""
    global _WORKER_LIKELIHOOD
    _WORKER_LIKELIHOOD = likelihood


def log_probability_worker(parameters: Sequence[float]) -> float:
    """Pickle-безпечна функція оцінювання для multiprocessing.Pool."""
    if _WORKER_LIKELIHOOD is None:
        raise RuntimeError("Робочий процес не отримав ініціалізовану модель.")
    return log_probability(parameters, _WORKER_LIKELIHOOD)


def initialize_walkers(
    nwalkers: int,
    model: str,
    seed: int,
) -> np.ndarray:
    """Готує незалежні стартові точки поблизу центру апріорного об'єму."""
    if model not in SAMPLED_PARAMETER_NAMES:
        raise ValueError(f"Невідома космологічна модель: {model}")
    ndim = len(SAMPLED_PARAMETER_NAMES[model])
    if nwalkers < 2 * ndim:
        raise ValueError("Для emcee потрібно щонайменше 2 × ndim спостерігачів.")
    rng = np.random.default_rng(seed)
    if model == "lcdm":
        center = np.array([70.0, 0.3, -19.3], dtype=np.float64)
        scale = np.array([1.0, 0.05, 0.1], dtype=np.float64)
    else:
        center = np.array([70.0, 0.3, -1.0, 0.0, -19.3], dtype=np.float64)
        scale = np.array([1.0, 0.05, 0.15, 0.25, 0.1], dtype=np.float64)
    positions = np.empty((nwalkers, ndim), dtype=np.float64)
    for walker in range(nwalkers):
        for _ in range(10_000):
            candidate = center + rng.normal(size=ndim) * scale
            if np.isfinite(log_prior(candidate, model)):
                positions[walker] = candidate
                break
        else:
            raise RuntimeError(f"Не вдалося ініціалізувати спостерігача {walker}.")
    if np.linalg.matrix_rank(positions - positions.mean(axis=0)) < ndim:
        raise RuntimeError("Початкові позиції спостерігачів вироджені.")
    return positions


def run_mcmc(
    likelihood: GaussianLikelihood,
    nwalkers: int,
    burn_in: int,
    steps: int,
    seed: int,
    processes: int | None = None,
    progress: bool = True,
) -> MCMCResult:
    """Проводить прогрів і production-запуск у spawn-сумісному пулі процесів."""
    ndim = len(likelihood.parameter_names)
    if nwalkers < 2 * ndim:
        raise ValueError(
            f"Для {ndim} параметрів emcee потрібно не менше {2 * ndim} спостерігачів."
        )
    if burn_in < 0 or steps <= 0:
        raise ValueError("burn-in має бути ≥ 0, а кількість кроків > 0.")

    available_cpus = os.cpu_count() or 1
    requested_processes = available_cpus if processes is None else processes
    if requested_processes < 1:
        raise ValueError("Кількість процесів має бути додатною.")
    # StretchMove оцінює одну половину ансамблю за раз, тому більш ніж
    # floor(nwalkers / 2) робочих процесів не отримали б незалежної роботи.
    worker_count = min(requested_processes, nwalkers // 2)
    if requested_processes > worker_count:
        LOGGER.warning(
            "Паралельно доступно не більше %s оцінок на крок; "
            "використаємо %s процесів.",
            nwalkers // 2,
            worker_count,
        )
    LOGGER.info(
        "Multiprocessing: %s процесів із доступних %s CPU.",
        worker_count,
        available_cpus,
    )

    initial = initialize_walkers(nwalkers, likelihood.model, seed)
    np.random.seed(seed)
    sampler: emcee.EnsembleSampler
    # spawn переносимий між Windows, macOS та Linux; глобальний масив даних
    # встановлюється top-level initializer'ом і не передається на кожен крок.
    context = multiprocessing.get_context("spawn")
    try:
        with context.Pool(
            processes=worker_count,
            initializer=initialize_likelihood_worker,
            initargs=(likelihood,),
        ) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability_worker,
                pool=pool,
            )
            state: object = initial
            if burn_in:
                LOGGER.info(
                    "Прогрів: %s кроків для кожного з %s спостерігачів.",
                    burn_in,
                    nwalkers,
                )
                state = sampler.run_mcmc(initial, burn_in, progress=progress)
                sampler.reset()
            LOGGER.info("Продукційна фаза: %s кроків на спостерігача.", steps)
            sampler.run_mcmc(state, steps, progress=progress)
    except (ValueError, RuntimeError, FloatingPointError, OSError) as exc:
        raise RuntimeError(f"MCMC-запуск перервано: {exc}") from exc

    samples = sampler.get_chain(flat=True)
    log_probabilities = sampler.get_log_prob(flat=True)
    maximum_a_posteriori = samples[int(np.argmax(log_probabilities))].copy()
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=np.float64)
    try:
        autocorrelation = np.asarray(sampler.get_autocorr_time(tol=0), dtype=np.float64)
    except Exception as exc:  # Діагностика не повинна приховати готовий ланцюг.
        LOGGER.warning("Не вдалося оцінити час автокореляції: %s", exc)
        autocorrelation = None
    if autocorrelation is not None:
        if steps < 50.0 * float(np.max(autocorrelation)):
            LOGGER.warning(
                "Ланцюг коротший за 50 найбільших часів автокореляції; "
                "збільште --steps і повторіть запуск перед науковою інтерпретацією."
            )
        else:
            LOGGER.info(
                "Оцінені часи автокореляції: %s",
                np.array2string(autocorrelation, precision=1),
            )
    LOGGER.info("Середня частка прийнятих кроків: %.3f", float(np.mean(acceptance)))
    if float(np.mean(acceptance)) < 0.1 or float(np.mean(acceptance)) > 0.8:
        LOGGER.warning(
            "Середня частка прийнятих кроків поза типовим робочим діапазоном; "
            "перевірте змішування ланцюгів та налаштування семплера."
        )
    return MCMCResult(
        samples=samples,
        maximum_a_posteriori=maximum_a_posteriori,
        acceptance_fraction=acceptance,
        autocorrelation_time=autocorrelation,
    )

def summarize_samples(
    samples: np.ndarray, parameter_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Оцінює медіану та асиметричні 68-відсоткові інтервали параметрів."""
    if (
        samples.ndim != 2
        or samples.shape[1] != len(parameter_names)
        or samples.shape[0] < 10
    ):
        raise ValueError("Недостатньо коректних апостеріорних вибірок.")
    quantiles = np.quantile(samples, (0.16, 0.50, 0.84), axis=0)
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(parameter_names):
        lower, median, upper = quantiles[:, index]
        result[name] = {
            "median": float(median),
            "lower_1sigma": float(lower),
            "upper_1sigma": float(upper),
            "minus": float(median - lower),
            "plus": float(upper - median),
        }
    return result


def save_results(
    output_directory: Path,
    result: MCMCResult,
    summary: dict[str, dict[str, float]],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Зберігає ланцюг NumPy та короткий JSON-звіт для відтворюваності."""
    model = str(metadata.get("model", "lcdm"))
    samples_path = output_directory / f"pantheon_posterior_samples_{model}.npz"
    summary_path = output_directory / f"pantheon_mcmc_summary_{model}.json"
    parameter_names = tuple(summary)
    try:
        np.savez_compressed(
            samples_path,
            samples=result.samples,
            parameter_names=np.asarray(parameter_names),
            model=np.asarray(model),
        )
        payload: dict[str, object] = {
            "parameters": summary,
            "maximum_a_posteriori": {
                name: float(result.maximum_a_posteriori[index])
                for index, name in enumerate(parameter_names)
            },
            "acceptance_fraction_mean": float(np.mean(result.acceptance_fraction)),
            "autocorrelation_time": (
                result.autocorrelation_time.tolist()
                if result.autocorrelation_time is not None
                else None
            ),
            **metadata,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"Не вдалося зберегти результати у {output_directory}: {exc}"
        ) from exc
    return samples_path, summary_path


def make_hubble_diagram(
    data: PantheonData,
    display_errors: np.ndarray,
    best_fit: np.ndarray,
    model: str,
    output_directory: Path,
) -> Path:
    """Створює двопанельну діаграму Габбла та залишків у високій роздільності."""
    h0, omega_m, w0, wa, absolute_magnitude = expand_parameters(best_fit, model)
    used = data.use_for_cosmology
    hubble_flow = ~data.is_calibrator[used]
    flow_redshift = data.redshift[used][hubble_flow]
    flow_redshift_hel = data.heliocentric_redshift[used][hubble_flow]
    flow_magnitude = data.apparent_magnitude[used][hubble_flow]
    flow_errors = display_errors[hubble_flow]
    curve_redshift = np.geomspace(
        float(np.min(flow_redshift)), float(np.max(flow_redshift)), 600
    )
    curve_mu = distance_modulus(
        curve_redshift,
        h0,
        omega_m,
        w0=w0,
        wa=wa,
        integration="quad",
    )
    model_at_data = distance_modulus(
        flow_redshift,
        h0,
        omega_m,
        w0=w0,
        wa=wa,
        redshift_hel=flow_redshift_hel,
        integration="gauss",
    )
    observed_mu = flow_magnitude - absolute_magnitude
    residuals = observed_mu - np.asarray(model_at_data)
    model_label = r"flat $\Lambda$CDM" if model == "lcdm" else r"flat $w_0w_a$CDM"

    figure, (axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": 0.04},
    )
    axis.errorbar(
        flow_redshift,
        observed_mu,
        yerr=flow_errors,
        fmt=".",
        markersize=3.0,
        elinewidth=0.45,
        capsize=0,
        color="#4C78A8",
        ecolor="#9BB7D4",
        alpha=0.58,
        rasterized=True,
        label="Pantheon+SH0ES Hubble-flow SNe",
    )
    axis.plot(
        curve_redshift,
        curve_mu,
        color="#D1495B",
        linewidth=2.0,
        label=(
            rf"Best-fit {model_label}: $H_0={h0:.2f}$, "
            rf"$\Omega_m={omega_m:.3f}$, $w_0={w0:.2f}$, "
            rf"$w_a={wa:.2f}$, $M_B={absolute_magnitude:.3f}$"
        ),
    )
    axis.set_xscale("log")
    axis.set_ylabel(r"Distance modulus $\mu$ (mag)")
    axis.set_title(rf"Pantheon+SH0ES: {model_label}")
    axis.legend(frameon=False, loc="best")
    axis.grid(True, which="both", alpha=0.18, linewidth=0.6)

    residual_axis.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    residual_axis.errorbar(
        flow_redshift,
        residuals,
        yerr=flow_errors,
        fmt=".",
        markersize=3.0,
        elinewidth=0.45,
        capsize=0,
        color="#4C78A8",
        ecolor="#9BB7D4",
        alpha=0.58,
        rasterized=True,
    )
    residual_axis.set_xscale("log")
    residual_axis.set_xlabel(r"Hubble-diagram redshift $z_{\mathrm{HD}}$")
    residual_axis.set_ylabel(r"$\Delta\mu$ (mag)")
    residual_axis.grid(True, which="both", alpha=0.18, linewidth=0.6)
    figure.tight_layout()

    output_path = output_directory / f"pantheon_hubble_diagram_{model}.png"
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    except OSError as exc:
        raise RuntimeError(f"Не вдалося зберегти рисунок {output_path}: {exc}") from exc
    finally:
        plt.close(figure)
    return output_path


def load_overlay_samples(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Завантажує власний NPZ-ланцюг для накладання контурів без pickle."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "samples" not in archive or "parameter_names" not in archive:
                raise ValueError(
                    "Файл накладання має містити samples та parameter_names."
                )
            samples = np.asarray(archive["samples"], dtype=np.float64)
            names = tuple(str(name) for name in archive["parameter_names"].tolist())
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Не вдалося прочитати ланцюг для контурів {path}: {exc}") from exc
    if samples.ndim != 2 or samples.shape[1] != len(names):
        raise ValueError("Розмірність ланцюга накладання не збігається з назвами.")
    return samples, names


def make_corner_plot(
    samples: np.ndarray,
    summary: dict[str, dict[str, float]],
    parameter_names: Sequence[str],
    model: str,
    output_directory: Path,
    overlay: tuple[np.ndarray, Sequence[str]] | None = None,
) -> Path:
    """Будує 2D/4D кутові апостеріорні контури, не показуючи M_B.

    За ``overlay`` можна передати ланцюг тієї самої параметричної розмірності;
    його 68% та 95% контури накладаються контрастним кольором.
    """
    contour_names = tuple(
        name for name in COSMOLOGICAL_CONTOUR_PARAMETERS if name in parameter_names
    )
    if len(contour_names) != (2 if model == "lcdm" else 4):
        raise ValueError("Набір космологічних параметрів не відповідає моделі.")
    indices = [parameter_names.index(name) for name in contour_names]
    plotted_samples = samples[:, indices]
    labels = [CONTOUR_PARAMETER_LABELS[name] for name in contour_names]
    medians = [summary[name]["median"] for name in contour_names]
    figure = corner.corner(
        plotted_samples,
        labels=labels,
        truths=medians,
        quantiles=(0.16, 0.50, 0.84),
        levels=(0.68, 0.95),
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 12},
        color="#315A7D",
        smooth=1.0,
        smooth1d=1.0,
        fill_contours=True,
        plot_density=True,
        plot_datapoints=False,
        max_n_ticks=4,
        use_math_text=True,
    )
    if overlay is not None:
        overlay_samples, overlay_names = overlay
        if tuple(name for name in contour_names if name in overlay_names) != contour_names:
            raise ValueError(
                "Для накладання потрібні ті самі космологічні параметри: "
                + ", ".join(contour_names)
            )
        overlay_indices = [overlay_names.index(name) for name in contour_names]
        corner.corner(
            overlay_samples[:, overlay_indices],
            labels=labels,
            fig=figure,
            levels=(0.68, 0.95),
            color="#D1495B",
            smooth=1.0,
            smooth1d=1.0,
            fill_contours=False,
            plot_density=False,
            plot_datapoints=False,
            show_titles=False,
            hist_kwargs={"histtype": "step", "linewidth": 1.2},
            contour_kwargs={"linewidths": 1.4},
            max_n_ticks=4,
            use_math_text=True,
        )
        from matplotlib.lines import Line2D

        figure.legend(
            handles=(
                Line2D([0], [0], color="#315A7D", lw=2, label="Поточна модель"),
                Line2D([0], [0], color="#D1495B", lw=1.5, label="Накладений ланцюг"),
            ),
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=False,
        )
    figure.suptitle(
        rf"Pantheon+SH0ES: апостеріорні розподіли ({model})",
        y=1.02,
        fontsize=13,
    )
    output_path = output_directory / f"pantheon_confidence_contours_{model}.png"
    try:
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    except OSError as exc:
        raise RuntimeError(f"Не вдалося зберегти рисунок {output_path}: {exc}") from exc
    finally:
        plt.close(figure)
    return output_path

def build_parser() -> argparse.ArgumentParser:
    """Створює інтерфейс командного рядка для відтворюваного запуску."""
    parser = argparse.ArgumentParser(
        description=(
            "MCMC-оцінювання H0 та Omega_m за Pantheon+SH0ES з повною "
            "матрицею коваріації STAT+SYS."
        )
    )
    parser.add_argument("--data", type=Path, help="Шлях до Pantheon+SH0ES.dat.")
    parser.add_argument(
        "--covariance",
        type=Path,
        help="Шлях до Pantheon+SH0ES_STAT+SYS.cov.",
    )
    parser.add_argument(
        "--model",
        choices=("lcdm", "w0wacdm"),
        default="lcdm",
        help="Космологічна модель: базова lcdm або динамічна w0wacdm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Каталог для рисунків, ланцюга та JSON-звіту.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=None,
        help=(
            "Кількість emcee-спостерігачів; типово не менше 32 і вдвічі "
            "більше числа доступних CPU для насичення пулу."
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Число процесів; за замовчуванням використати доступні CPU.",
    )
    parser.add_argument(
        "--burn-in", type=int, default=2000, help="Кроки фази прогріву."
    )
    parser.add_argument(
        "--steps", type=int, default=5000, help="Кроки продукційної фази."
    )
    parser.add_argument(
        "--seed", type=int, default=20261024, help="Початкове значення генератора."
    )
    parser.add_argument(
        "--integration-order",
        type=int,
        default=64,
        help="Порядок квадратури Гаусса—Лежандра (не менше 16).",
    )
    parser.add_argument(
        "--allow-pseudoinverse",
        action="store_true",
        help="Дозволити псевдообернення лише для виродженої PSD-матриці.",
    )
    parser.add_argument("--verbose", action="store_true", help="Докладні повідомлення.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Не показувати лічильник прогресу emcee.",
    )
    parser.add_argument(
        "--overlay-samples",
        type=Path,
        help=(
            "NPZ-файл попереднього запуску з тими самими космологічними "
            "параметрами для накладання контурів."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запускає повний конвеєр обробки, MCMC, збереження та візуалізації."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.walkers is None:
        args.walkers = max(32, 2 * (os.cpu_count() or 1))

    try:
        data_path = resolve_release_file("Pantheon+SH0ES.dat", args.data)
        covariance_path = resolve_release_file(
            "Pantheon+SH0ES_STAT+SYS.cov", args.covariance
        )
        data = load_pantheon_data(data_path)
        full_covariance = load_covariance_matrix(
            covariance_path, expected_size=data.redshift.size
        )
        selection = data.use_for_cosmology
        covariance = full_covariance[np.ix_(selection, selection)]
        metric = prepare_covariance(
            covariance, allow_pseudoinverse=args.allow_pseudoinverse
        )
        integration_difference = validate_quadrature(
            args.integration_order,
            float(np.max(data.redshift[selection & ~data.is_calibrator])),
            model=args.model,
        )
        LOGGER.info(
            "Завантажено %s рядків; у космологічній функції правдоподібності %s.",
            data.redshift.size,
            int(np.count_nonzero(selection)),
        )
        LOGGER.info("Метод коваріації: %s.", metric.method)
        LOGGER.info("log|C| = %.8f; ранг = %s.", metric.log_determinant, metric.rank)
        LOGGER.info(
            "Максимальне відхилення квадратури від quad: %.3e mag.",
            integration_difference,
        )

        parameter_names = SAMPLED_PARAMETER_NAMES[args.model]
        likelihood = GaussianLikelihood(
            data,
            metric,
            args.integration_order,
            model=args.model,
        )
        result = run_mcmc(
            likelihood,
            nwalkers=args.walkers,
            burn_in=args.burn_in,
            steps=args.steps,
            seed=args.seed,
            processes=args.processes,
            progress=not args.no_progress,
        )
        summary = summarize_samples(result.samples, parameter_names)
        chi_squared_map = likelihood.chi_squared(result.maximum_a_posteriori)
        full_map = expand_parameters(result.maximum_a_posteriori, args.model)
        degrees_of_freedom = metric.rank - len(parameter_names)
        if degrees_of_freedom <= 0:
            raise ValueError(
                "Кількість ступенів свободи має бути додатною після оцінки параметрів."
            )
        reduced_chi_squared = chi_squared_map / degrees_of_freedom
        LOGGER.info(
            "χ²_min у найкращій MAP-вибірці = %.3f; dof = %s; χ²_red = %.5f.",
            chi_squared_map,
            degrees_of_freedom,
            reduced_chi_squared,
        )
        units = {
            "H0": "km s^-1 Mpc^-1",
            "Omega_m": "",
            "w0": "",
            "wa": "",
            "M_B": "mag",
        }
        for name in parameter_names:
            item = summary[name]
            LOGGER.info(
                "%s = %.5f +%.5f / -%.5f %s (68%% equal-tail)",
                name,
                item["median"],
                item["plus"],
                item["minus"],
                units[name],
            )

        output_directory = args.output_dir.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        display_errors = np.sqrt(np.diag(covariance))
        hubble_path = make_hubble_diagram(
            data,
            display_errors,
            result.maximum_a_posteriori,
            args.model,
            output_directory,
        )
        overlay = (
            load_overlay_samples(args.overlay_samples)
            if args.overlay_samples is not None
            else None
        )
        corner_path = make_corner_plot(
            result.samples,
            summary,
            parameter_names,
            args.model,
            output_directory,
            overlay=overlay,
        )
        metadata: dict[str, object] = {
            "model": args.model,
            "data_file": str(data_path),
            "covariance_file": str(covariance_path),
            "number_of_rows": int(data.redshift.size),
            "number_of_rows_used": int(np.count_nonzero(selection)),
            "number_of_unique_supernova_names": int(pd.Series(data.names).nunique()),
            "number_of_cepheid_calibrators": int(
                np.count_nonzero(data.is_calibrator & selection)
            ),
            "number_of_hubble_flow_rows": int(
                np.count_nonzero(selection & ~data.is_calibrator)
            ),
            "maximum_a_posteriori": {
                name: float(result.maximum_a_posteriori[index])
                for index, name in enumerate(parameter_names)
            },
            "maximum_a_posteriori_full_order": {
                name: float(full_map[index])
                for index, name in enumerate(FULL_PARAMETER_NAMES)
            },
            "chi_squared_min_at_sampled_map": chi_squared_map,
            "degrees_of_freedom": degrees_of_freedom,
            "reduced_chi_squared_at_sampled_map": reduced_chi_squared,
            "covariance_method": metric.method,
            "covariance_log_determinant": metric.log_determinant,
            "covariance_rank": metric.rank,
            "integration_order": args.integration_order,
            "quadrature_vs_quad_max_abs_mag": integration_difference,
            "walkers": args.walkers,
            "processes": min(
                args.processes or (os.cpu_count() or 1), args.walkers // 2
            ),
            "burn_in_steps": args.burn_in,
            "production_steps": args.steps,
            "random_seed": args.seed,
            "prior": {
                "H0": [50.0, 90.0],
                "Omega_m": [0.0, 1.0],
                "M_B": [-20.5, -18.0],
                "w0": [-3.0, 1.0] if args.model == "w0wacdm" else -1.0,
                "wa": [-3.0, 3.0] if args.model == "w0wacdm" else 0.0,
            },
            "sampled_parameter_order": list(parameter_names),
            "physical_parameter_order": list(FULL_PARAMETER_NAMES),
            "data_vector": "m_b_corr with CEPH_DIST anchors",
            "error_bars_in_figures": "sqrt(diag(full STAT+SYS covariance))",
            "hubble_diagram": str(hubble_path),
            "corner_plot": str(corner_path),
        }
        samples_path, summary_path = save_results(
            output_directory, result, summary, metadata
        )
        LOGGER.info("Рисунок Габбла: %s", hubble_path)
        LOGGER.info("Контури параметрів без M_B: %s", corner_path)
        LOGGER.info("Апостеріорні вибірки: %s", samples_path)
        LOGGER.info("Підсумок та параметри запуску: %s", summary_path)
        return 0
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        LOGGER.error("Аналіз завершився з помилкою: %s", exc)
        return 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
