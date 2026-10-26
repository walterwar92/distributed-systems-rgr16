"""
Аналитическое решение РГР, вариант 16 (этап 2).

Стационарная температура в цилиндре радиуса a, высоты h, торцы
теплоизолированы, на боковой поверхности T = 0 при z < c и T = T0 при z >= c:

    T(r,z) = T0 (1 - c/h)
             - (2 T0 / pi) * sum_{n>=1} sin(n pi c/h) / n
                           * I0(n pi r/h) / I0(n pi a/h) * cos(n pi z/h).

Скрипт
  * проверяет выкладки главы 3 (вронскиан I_m, K_m; скачок радиальной
    функции Грина; симметрию G; граничные условия; уравнение Лапласа);
  * печатает таблицу сходимости ряда;
  * строит рисунки для отчёта в report/figures/.

Запуск из корня репозитория:  python code/analytic.py
Нужны numpy, scipy, matplotlib. Функции Бесселя из scipy используются
только для аналитической формулы; на этапе 3 (метод сеток) готовые
решатели не применяются.
"""

from pathlib import Path

import numpy as np
from scipy.special import i0e, ive, kve

FIG = Path(__file__).resolve().parent.parent / "report" / "figures"


# ------------------------------------------------------------------
# Аналитическое решение
# ------------------------------------------------------------------
def bessel_ratio(n, r, a, h):
    """I0(k r) / I0(k a), k = n pi / h, без переполнения:
    I0(x) = i0e(x) e^x, поэтому отношение = i0e(kr)/i0e(ka) * e^{k(r-a)}."""
    k = n * np.pi / h
    return i0e(k * r) / i0e(k * a) * np.exp(k * (r - a))


def temperature(r, z, a=0.5, h=1.0, c=0.4, T0=1.0, N=400):
    """Частичная сумма ряда (N слагаемых с n >= 1)."""
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    T = np.full(np.broadcast(r, z).shape, T0 * (1 - c / h))
    for n in range(1, N + 1):
        T = T - 2 * T0 / np.pi * np.sin(n * np.pi * c / h) / n \
            * bessel_ratio(n, r, a, h) * np.cos(n * np.pi * z / h)
    return T


# ------------------------------------------------------------------
# Функция Грина (полный трёхмерный ряд) — для проверок
# ------------------------------------------------------------------
def g_radial(m, n, r, r0, a, h):
    """Радиальная функция Грина g_mn(r, r0)."""
    rl, rg = min(r, r0), max(r, r0)
    if n == 0:
        if m == 0:
            return np.log(a / rg)
        return ((rl / rg) ** m - (rl * rg / a**2) ** m) / (2 * m)
    k = n * np.pi / h
    # I_m(k r<) [K_m(k r>) - I_m(k r>) K_m(k a) / I_m(k a)]
    # через масштабированные функции: I = ive e^x, K = kve e^-x
    t1 = ive(m, k * rl) * kve(m, k * rg) * np.exp(k * (rl - rg))
    t2 = (ive(m, k * rl) * ive(m, k * rg) * kve(m, k * a) / ive(m, k * a)
          * np.exp(k * (rl + rg - 2 * a)))
    return t1 - t2


def green(M, M0, a, h, mmax=40, nmax=200):
    r, phi, z = M
    r0, phi0, z0 = M0
    s = 0.0
    for m in range(mmax + 1):
        em = 1 if m == 0 else 2
        for n in range(nmax + 1):
            en = 1 if n == 0 else 2
            s += (em * en / (2 * np.pi * h) * np.cos(m * (phi - phi0))
                  * np.cos(n * np.pi * z / h) * np.cos(n * np.pi * z0 / h)
                  * g_radial(m, n, r, r0, a, h))
    return s


def checks():
    a, h, c = 0.5, 1.0, 0.4
    print("=== Проверки выкладок ===")

    # 1) Вронскиан  I_m K_m' - I_m' K_m = -1/x
    from scipy.special import iv, kv, ivp, kvp
    for m in (0, 1, 3):
        for x in (0.3, 2.0, 7.0):
            w = iv(m, x) * kvp(m, x) - ivp(m, x) * kv(m, x)
            assert abs(w * x + 1) < 1e-10, (m, x, w)
    print("вронскиан I_m K_m' - I_m' K_m = -1/x: ок")

    # 2) Скачок производной радиальной функции Грина: r0 [g']_{r0} = -1
    eps = 1e-6
    for m, n in ((0, 0), (2, 0), (0, 3), (2, 5)):
        r0 = 0.3
        dp = (g_radial(m, n, r0 + 2 * eps, r0, a, h)
              - g_radial(m, n, r0 + eps, r0, a, h)) / eps
        dm = (g_radial(m, n, r0 - eps, r0, a, h)
              - g_radial(m, n, r0 - 2 * eps, r0, a, h)) / eps
        assert abs(r0 * (dp - dm) + 1) < 1e-4, (m, n, r0 * (dp - dm))
        assert abs(g_radial(m, n, a, r0, a, h)) < 1e-12
    print("скачок r0[g']=-1 и g(a)=0: ок")

    # 3) Симметрия G(M, M0) = G(M0, M) и G = 0 на боковой поверхности
    M, M0 = (0.2, 0.7, 0.3), (0.35, 2.0, 0.8)
    g1, g2 = green(M, M0, a, h), green(M0, M, a, h)
    assert abs(g1 - g2) < 1e-10 * abs(g1)
    assert abs(green((a, 1.0, 0.5), M0, a, h)) < 1e-12
    # Вблизи M0 функция Грина ведёт себя как 1/(4 pi R)
    d = 0.02
    Mn = (M0[0] + d, M0[1], M0[2])
    print(f"G(M,M0)=G(M0,M)={g1:.6f}; G*4*pi*R при R={d}: "
          f"{green(Mn, M0, a, h, mmax=150, nmax=600) * 4 * np.pi * d:.3f}")

    # 4) Граничные условия и уравнение решения
    z = np.linspace(0.01, 0.99, 50)
    Tw = temperature(a, z, a, h, c, N=20000)
    mu = np.where(z >= c, 1.0, 0.0)
    far = np.abs(z - c) > 0.05
    print(f"|T(a,z) - mu(z)| вдали от z=c: {np.max(np.abs(Tw - mu)[far]):.1e}")
    print(f"T(a,c) = {temperature(a, c, a, h, c, N=20000):.4f} (ожидается 0.5)")
    dz = 1e-4
    for r in (0.0, 0.25, 0.45):
        for zz in (0.0, h):
            sgn = 1 if zz == 0 else -1
            der = (temperature(r, zz + sgn * dz, a, h, c)
                   - temperature(r, zz, a, h, c)) / dz
            assert abs(der) < 1e-3, (r, zz, der)
    print("dT/dz = 0 на торцах: ок")
    hs = 1e-3
    for r, zz in ((0.1, 0.3), (0.3, 0.45), (0.4, 0.9)):
        Tc = temperature(r, zz, a, h, c)
        lap = ((temperature(r + hs, zz, a, h, c) - 2 * Tc
                + temperature(r - hs, zz, a, h, c)) / hs**2
               + (temperature(r + hs, zz, a, h, c)
                  - temperature(r - hs, zz, a, h, c)) / (2 * hs) / r
               + (temperature(r, zz + hs, a, h, c) - 2 * Tc
                  + temperature(r, zz - hs, a, h, c)) / hs**2)
        assert abs(lap) < 1e-3, (r, zz, lap)
    print("T_rr + T_r/r + T_zz = 0 внутри: ок")

    # 5) Решение через функцию Грина = ряд: численно интегрируем
    #    -a * dG/dr(a) * mu по боковой поверхности
    r0, z0 = 0.3, 0.55
    zq = np.linspace(c, h, 4001)
    phq = np.linspace(0, 2 * np.pi, 9)[:-1]
    eps = 1e-6
    vals = []
    for zz in zq[::40]:
        s = 0.0
        for p in phq:
            dG = (green((a, p, zz), (r0, 0.0, z0), a, h, mmax=3, nmax=200)
                  - green((a - eps, p, zz), (r0, 0.0, z0), a, h,
                          mmax=3, nmax=200)) / eps
            s += dG
        vals.append(-a * s * 2 * np.pi / len(phq))
    Tg = np.trapezoid(vals, zq[::40])
    print(f"T через интеграл с dG/dn: {Tg:.4f}; по формуле ряда: "
          f"{temperature(r0, z0, a, h, c):.4f}")

    # 6) Средняя по высоте температура не зависит от r
    zz = np.linspace(0, h, 20001)
    for r in (0.0, 0.25, 0.49):
        print(f"(1/h)∫T dz при r={r}: "
              f"{np.trapezoid(temperature(r, zz, a, h, c), zz) / h:.5f}"
              f" (1-c/h = {1 - c / h})")


def convergence_table():
    a, h, c = 0.5, 1.0, 0.4
    pts = [(0.0, c), (0.25, c), (0.45, c), (0.25, 0.1), (0.25, 0.9)]
    Ns = (1, 2, 5, 10, 20, 50)
    print("\n=== Сходимость: theta(r, z) для a/h=0.5, c/h=0.4 ===")
    print("r/a   z/h   " + "  ".join(f"N={N:<5d}" for N in Ns))
    for r, z in pts:
        row = [temperature(r, z, a, h, c, N=N) for N in Ns]
        print(f"{r / a:4.2f}  {z / h:4.2f}  " +
              "  ".join(f"{v:.5f}" for v in row))
    # максимум модуля n-го члена на окружности r = r при разных r
    print("\nМодуль 21-го члена ряда, 2/(pi n)|sin| I0(kr)/I0(ka):")
    for r in (0.0, 0.25, 0.45, 0.5):
        n = 21
        t = 2 / (np.pi * n) * abs(np.sin(n * np.pi * c / h)) \
            * bessel_ratio(n, r, a, h)
        print(f"  r/a = {r / a:.1f}: {t:.2e}")


# ------------------------------------------------------------------
# Рисунки
# ------------------------------------------------------------------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.unicode_minus": True,
        "axes.formatter.use_locale": False,
    })
    a, h, c = 0.5, 1.0, 0.4

    def comma(ax):
        from matplotlib.ticker import FuncFormatter
        f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
        ax.xaxis.set_major_formatter(f)
        ax.yaxis.set_major_formatter(f)

    # --- Рисунок: изотермы в осевом сечении ---
    r = np.linspace(-a, a, 301)
    z = np.linspace(0, h, 601)
    R, Z = np.meshgrid(r, z)
    TH = temperature(np.abs(R), Z, a, h, c, N=600)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cf = ax.contourf(R / a, Z / h, TH, levels=np.linspace(0, 1, 21),
                     cmap="coolwarm")
    cs = ax.contour(R / a, Z / h, TH, levels=np.arange(0.1, 1.0, 0.1),
                    colors="k", linewidths=0.6)
    ax.clabel(cs, fmt=lambda v: f"{v:.1f}".replace(".", ","), fontsize=9)
    ax.axvline(0, color="k", lw=0.5, ls="-.")
    ax.axhline(c / h, xmin=0, xmax=0.04, color="k", lw=1.5)
    ax.axhline(c / h, xmin=0.96, xmax=1, color="k", lw=1.5)
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$z/h$")
    ax.set_aspect(h / a)   # истинные пропорции сечения
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    cb = fig.colorbar(cf, ax=ax, ticks=np.linspace(0, 1, 6), shrink=0.95)
    cb.set_label(r"$T/T_0$")
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    comma(ax)
    ax.set_xticklabels(["1", "0,5", "0", "0,5", "1"])
    fig.tight_layout()
    fig.savefig(FIG / "fig_isotherms.pdf")
    plt.close(fig)

    # --- Рисунок: частичные суммы на боковой поверхности ---
    z = np.linspace(0, h, 4001)
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    mu = np.where(z >= c, 1.0, 0.0)
    ax.plot(z / h, mu, color="k", lw=2.2, label=r"$\mu(z)/T_0$")
    for N, st in ((3, ":"), (10, "--"), (60, "-")):
        ax.plot(z / h, temperature(a, z, a, h, c, N=N), st, lw=1.0,
                label=f"$N={N}$")
    ax.set_xlabel(r"$z/h$")
    ax.set_ylabel(r"$T(a,z)/T_0$")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.grid(lw=0.3)
    comma(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_partial_sums.pdf")
    plt.close(fig)

    # --- Рисунок: профили T(z) при разных r ---
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for rr, st in ((0.0, "-"), (0.5, "--"), (0.8, "-."), (0.95, ":")):
        ax.plot(z / h, temperature(rr * a, z, a, h, c, N=600), st, lw=1.2,
                label=f"$r/a={rr:g}$".replace(".", "{,}"))
    ax.plot(z / h, mu, color="0.5", lw=0.8,
            label=r"$r=a$")
    ax.axhline(1 - c / h, color="k", lw=0.5)
    ax.text(0.02, 1 - c / h + 0.02, r"$1-c/h$")
    ax.set_xlabel(r"$z/h$")
    ax.set_ylabel(r"$T/T_0$")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.grid(lw=0.3)
    comma(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_profiles.pdf")
    plt.close(fig)
    print("\nрисунки сохранены в", FIG)


if __name__ == "__main__":
    checks()
    convergence_table()
    figures()
