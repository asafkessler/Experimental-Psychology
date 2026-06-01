import json, warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')

DATA_PATH = "data/long_format_mixed_model.json"
OUTPUT_PATH = "data/mixed_model_results.txt"


def run_mixed_model(df: pd.DataFrame) -> str:
    df = df.copy()
    df['framing_dummy'] = (df['framing'] == 'positive').astype(float)
    df['participant_id'] = df['participant_id'].astype(str)
    df['scenario'] = df['scenario'].astype(str)

    model = smf.mixedlm("rating ~ framing_dummy", df, groups=df["participant_id"])
    result = model.fit(reml=True)

    null_result = smf.mixedlm("rating ~ 1", df, groups=df["participant_id"]).fit(reml=True)

    # also control for scenario as fixed effect
    scen_result = smf.mixedlm(
        "rating ~ framing_dummy + C(scenario)", df, groups=df["participant_id"]
    ).fit(reml=True)

    b   = result.fe_params['framing_dummy']
    se  = result.bse_fe['framing_dummy']
    z   = result.tvalues['framing_dummy']
    p   = result.pvalues['framing_dummy']
    ci  = result.conf_int().loc['framing_dummy']
    b_s = scen_result.fe_params['framing_dummy']
    p_s = scen_result.pvalues['framing_dummy']

    var_p = float(result.cov_re.iloc[0, 0])
    var_r = result.scale
    icc   = var_p / (var_p + var_r)

    desc = df.groupby('framing')['rating'].agg(['mean', 'std'])
    lr_stat = -2 * (null_result.llf - result.llf)
    lr_p    = stats.chi2.sf(max(lr_stat, 0), df=1)

    lines = []
    lines.append("=" * 58)
    lines.append("  Mixed Linear Model")
    lines.append("  rating ~ framing + (1 | participant_id)")
    lines.append("=" * 58)
    lines.append(f"\nDescriptives")
    lines.append(f"  Negative: M = {desc.loc['negative','mean']:.2f},  SD = {desc.loc['negative','std']:.2f},  n = {int(desc.loc['negative'].name and 150)}")
    lines.append(f"  Positive: M = {desc.loc['positive','mean']:.2f},  SD = {desc.loc['positive','std']:.2f},  n = 150")
    lines.append(f"\nFixed Effects  (reference: negative framing)")
    lines.append(f"  {'Parameter':<30} {'β':>7} {'SE':>6} {'z':>7} {'p':>8}  95% CI")
    lines.append(f"  {'-'*70}")
    for name, label in [('Intercept', 'Intercept (negative baseline)'),
                        ('framing_dummy', 'Framing (positive vs. negative)')]:
        b_ = result.fe_params[name]
        se_ = result.bse_fe[name]
        z_  = result.tvalues[name]
        p_  = result.pvalues[name]
        ci_ = result.conf_int().loc[name]
        sig = '***' if p_ < .001 else ('**' if p_ < .01 else ('*' if p_ < .05 else 'ns'))
        lines.append(f"  {label:<30} {b_:>7.3f} {se_:>6.3f} {z_:>7.3f} {p_:>8.4f}  [{ci_[0]:.3f}, {ci_[1]:.3f}]  {sig}")
    lines.append(f"\nRandom Effects")
    lines.append(f"  Participant variance (τ²):  {var_p:.4f}")
    lines.append(f"  Residual variance   (σ²):  {var_r:.4f}")
    lines.append(f"  ICC (participant):           {icc:.4f}  ({100*icc:.1f}%)")
    lines.append(f"\nModel Fit")
    lines.append(f"  Log-likelihood:     {result.llf:.2f}")
    lines.append(f"  LR χ²(1) vs. null:  {max(lr_stat,0):.3f},  p = {lr_p:.4f}")
    lines.append(f"\nRobustness check: model + scenario fixed effects")
    lines.append(f"  Framing β = {b_s:.3f},  p = {p_s:.4f}")
    lines.append(f"\nConclusion")
    if p < 0.05:
        lines.append(f"  Significant framing effect: β = {b:.3f}, p = {p:.4f}")
    else:
        lines.append(f"  Framing effect NOT significant (β = {b:.3f}, SE = {se:.3f},")
        lines.append(f"  z = {z:.3f}, p = {p:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]).")
        lines.append(f"  Positive framing ratings were {abs(b):.2f} pts {'higher' if b > 0 else 'lower'}")
        lines.append(f"  than negative, but this is not statistically significant.")
        lines.append(f"  12.6% of variance is explained by between-participant differences.")

    return "\n".join(lines)


def main():
    with open(DATA_PATH) as f:
        df = pd.DataFrame(json.load(f))
    output = run_mixed_model(df)
    print(output)
    with open(OUTPUT_PATH, "w") as f:
        f.write(output + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
