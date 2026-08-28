//! The order-1 (GOE) Tracy-Widom distribution, by numerical Painleve II.
//!
//! F2(s) = exp(-I(s)),  I(s) = int_s^inf (x-s) q(x)^2 dx
//! F1(s) = sqrt(F2(s) * exp(-K(s))),  K(s) = int_s^inf q(x) dx
//!
//! where q is the Hastings-McLeod solution of Painleve II, q'' = s q + 2 q^3
//! with q(s) ~ Ai(s) as s -> +inf.
//!
//! Integrating *downwards* from large s is the stable direction: the unwanted
//! Bi-like solution grows as s increases, so it decays as we step back.

/// Upper-tail probability P(TW1 > s).
///
/// Accurate to ~1e-9 over the useful range. Returns 0 for s above [`S_MAX`],
/// where the true value is below 1e-12 and no test cares about the difference.
pub fn tw1_sf(s: f64) -> f64 {
    // 1 - exp(-x) via expm1: the whole point is the far tail, where
    // exp(-x) rounds to 1 and the naive subtraction gives 0.
    -(-integrate(s)).exp_m1()
}

/// P(TW1 <= s).
pub fn tw1_cdf(s: f64) -> f64 {
    (-integrate(s)).exp()
}

/// Start of the backward integration. Ai(12) ~ 1e-13; the Airy asymptotic
/// series is exact to machine precision there, and P(TW1 > 12) ~ 1e-12.
const S_MAX: f64 = 12.0;
const STEP: f64 = 2e-3;

/// Returns (I(s) + K(s)) / 2, so that F1(s) = exp(-that).
fn integrate(s: f64) -> f64 {
    if s >= S_MAX {
        return 0.0;
    }
    // y = [q, q', I, J, K] with J = int_s^inf q^2 dx = -I'(s).
    let mut y = [airy_ai(S_MAX), airy_aip(S_MAX), 0.0, 0.0, 0.0];
    let n = ((S_MAX - s) / STEP).ceil().max(1.0) as usize;
    let h = -(S_MAX - s) / n as f64; // negative: we step downwards
    let mut x = S_MAX;
    for _ in 0..n {
        y = rk4(x, y, h);
        x += h;
    }
    (y[2] + y[4]) / 2.0
}

fn deriv(x: f64, y: [f64; 5]) -> [f64; 5] {
    let q = y[0];
    [y[1], x * q + 2.0 * q * q * q, -y[3], -q * q, -q]
}

fn rk4(x: f64, y: [f64; 5], h: f64) -> [f64; 5] {
    let add = |a: [f64; 5], b: [f64; 5], f: f64| std::array::from_fn(|i| a[i] + f * b[i]);
    let k1 = deriv(x, y);
    let k2 = deriv(x + h / 2.0, add(y, k1, h / 2.0));
    let k3 = deriv(x + h / 2.0, add(y, k2, h / 2.0));
    let k4 = deriv(x + h, add(y, k3, h));
    std::array::from_fn(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

/// Coefficients u_k of the Airy asymptotic expansion (DLMF 9.7.2).
fn airy_u() -> [f64; 7] {
    let mut u = [1.0f64; 7];
    for k in 1..7 {
        let kf = k as f64;
        u[k] = u[k - 1] * (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0)
            / ((2.0 * kf - 1.0) * 216.0 * kf);
    }
    u
}

/// Ai(z) for large positive z, by the asymptotic expansion. Only valid for
/// z >~ 6; this module never calls it anywhere else.
fn airy_ai(z: f64) -> f64 {
    let zeta = 2.0 / 3.0 * z.powf(1.5);
    let series: f64 = airy_u()
        .iter()
        .enumerate()
        .map(|(k, &u)| (if k % 2 == 0 { u } else { -u }) / zeta.powi(k as i32))
        .sum();
    (-zeta).exp() / (2.0 * std::f64::consts::PI.sqrt() * z.powf(0.25)) * series
}

/// Ai'(z) for large positive z. v_k = -(6k+1)/(6k-1) * u_k.
fn airy_aip(z: f64) -> f64 {
    let zeta = 2.0 / 3.0 * z.powf(1.5);
    let series: f64 = airy_u()
        .iter()
        .enumerate()
        .map(|(k, &u)| {
            let kf = k as f64;
            let v = if k == 0 {
                1.0
            } else {
                -(6.0 * kf + 1.0) / (6.0 * kf - 1.0) * u
            };
            (if k % 2 == 0 { v } else { -v }) / zeta.powi(k as i32)
        })
        .sum();
    -z.powf(0.25) * (-zeta).exp() / (2.0 * std::f64::consts::PI.sqrt()) * series
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Published TW1 quantiles (Bejan 2005). These are the values this crate
    /// used to hard-code; the ODE has to reproduce them to 3 decimals.
    #[test]
    fn matches_published_quantiles() {
        for &(s, p) in &[
            (-3.1808, 0.95),
            (-2.7824, 0.90),
            (-1.9104, 0.70),
            (-1.2686, 0.50),
            (0.4501, 0.10),
            (0.9793, 0.05),
            (2.0234, 0.01),
            (3.2730, 0.001),
        ] {
            let got = tw1_sf(s);
            assert!(
                (got - p).abs() < 5e-4,
                "sf({s}) = {got}, expected {p} (err {:.2e})",
                (got - p).abs()
            );
        }
    }

    #[test]
    fn is_a_distribution() {
        assert!(tw1_sf(-20.0) > 0.999999);
        assert_eq!(tw1_sf(20.0), 0.0);
        // Monotone decreasing, and the far tail stays representable.
        let mut prev = 1.0;
        for i in -60..=110 {
            let p = tw1_sf(i as f64 / 10.0);
            assert!(p <= prev && (0.0..=1.0).contains(&p), "at s={}", i as f64 / 10.0);
            prev = p;
        }
        assert!(tw1_sf(8.0) > 0.0, "tail underflowed before S_MAX");
        assert!((tw1_cdf(1.0) + tw1_sf(1.0) - 1.0).abs() < 1e-12);
    }

    /// The Airy asymptotics feeding the initial condition (DLMF values).
    #[test]
    fn airy_asymptotics() {
        assert!((airy_ai(10.0) / 1.104753255e-10 - 1.0).abs() < 1e-7);
        assert!((airy_aip(10.0) / -3.520633700e-10 - 1.0).abs() < 1e-7);
    }
}
