import math

import numpy as np
import pytest
import scipy.stats as ss

from .context import core


class TestMisc:
    def test_observed_lift_relative(self):
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.1

        actual = core.observed_lift(trials, successes, lift="relative")
        assert actual == pytest.approx(expected)

        actual = core.observed_lift(trials, successes)
        assert actual == pytest.approx(expected)

    def test_observed_lift(self):
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.01
        actual = core.observed_lift(trials, successes, lift="absolute")
        assert actual == pytest.approx(expected)

    def test_observed_lift_undefined(self):
        trials = [1000, 1000]
        successes = [0, 1]
        with pytest.raises(ZeroDivisionError):
            core.observed_lift(trials, successes)

    @pytest.mark.parametrize(
        "group_sizes,baseline,null_lift,alt_lift,expected_p_alt",
        [
            ([1000, 1000], 0.10, 0.0, 0.10, [0.09502262, 0.10452489]),
            ([1000, 1000], 0.10, 0.0, -0.10, [0.10497238, 0.09447514]),
            ([1000, 1000], 0.10, 0.10, 0.20, [0.09525275, 0.1143033]),
            ([200, 1800], 0.10, 0.0, 0.10, [0.09167368, 0.10084104]),
            ([1000, 1000], 0.0001, 0.0, 0.10, [9.50226256e-05, 1.04524886e-04]),
        ],
    )
    def test_simple_hypothesis_from_composite_relative_lift(
        self, group_sizes, baseline, null_lift, alt_lift, expected_p_alt
    ):
        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency.
        #
        # na = group_sizes[0]
        # nb = group_sizes[1]
        # p_null_a = baseline
        # p_null_b = p_null_a * (1 + null_lift)
        # p_alt = cp.Variable(2)
        # objective = cp.Minimize(
        #     (na / (p_null_a * (1 - p_null_a))) * (p_alt[0] - p_null_a) ** 2
        #     + (nb / (p_null_b * (1 - p_null_b))) * (p_alt[1] - p_null_b) ** 2
        # )
        # constraints = [0 <= p_alt, p_alt <= 1, p_alt[1] == p_alt[0] * (1 + alt_lift)]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected_p_alt = p_alt.value

        actual_p_null, actual_p_alt = core.simple_hypothesis_from_composite(
            group_sizes, baseline, null_lift, alt_lift, lift="relative"
        )

        assert actual_p_null[0] == baseline
        assert actual_p_null[1] == (1 + null_lift) * actual_p_null[0]

        assert actual_p_alt[1] == (1 + alt_lift) * actual_p_alt[0]
        assert actual_p_alt[0] >= 0.0
        assert actual_p_alt[0] <= 1.0
        assert actual_p_alt[1] >= 0.0
        assert actual_p_alt[1] <= 1.0

        assert actual_p_alt == pytest.approx(expected_p_alt)

    @pytest.mark.parametrize(
        "group_sizes,baseline,null_lift,alt_lift,expected_p_alt",
        [
            ([1000, 1000], 0.10, 0.0, 0.10, [0.05, 0.15]),
            ([1000, 1000], 0.10, 0.0, -0.10, [0.15, 0.05]),
            ([1000, 1000], 0.10, 0.10, 0.20, [0.064, 0.264]),
            ([200, 1800], 0.10, 0.0, 0.10, [0.01, 0.11]),
            ([1000, 1000], 0.0001, 0.0, 0.00005, [0.000075, 0.000125]),
        ],
    )
    def test_simple_hypothesis_from_composite_absolute_lift(
        self, group_sizes, baseline, null_lift, alt_lift, expected_p_alt
    ):
        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency.
        #
        # na = group_sizes[0]
        # nb = group_sizes[1]
        # p_null_a = baseline
        # p_null_b = p_null_a + null_lift
        # p_alt = cp.Variable(2)
        # objective = cp.Minimize(
        #     (na / (p_null_a * (1 - p_null_a))) * (p_alt[0] - p_null_a) ** 2
        #     + (nb / (p_null_b * (1 - p_null_b))) * (p_alt[1] - p_null_b) ** 2
        # )
        # constraints = [0 <= p_alt, p_alt <= 1, p_alt[1] == p_alt[0] + alt_lift]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected_p_alt = p_alt.value

        actual_p_null, actual_p_alt = core.simple_hypothesis_from_composite(
            group_sizes, baseline, null_lift, alt_lift, lift="absolute"
        )

        assert actual_p_null[0] == baseline
        assert actual_p_null[1] == actual_p_null[0] + null_lift

        assert actual_p_alt[1] == actual_p_alt[0] + alt_lift
        assert actual_p_alt[0] >= 0.0
        assert actual_p_alt[0] <= 1.0
        assert actual_p_alt[1] >= 0.0
        assert actual_p_alt[1] <= 1.0

        assert actual_p_alt == pytest.approx(expected_p_alt)

    def test_wilson_interval(self):
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.08290944359309571
        expected_high = 0.1201519631953484

        actual_low, actual_high = core.wilson_interval(s, n, alpha=alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @pytest.mark.parametrize(
        "alpha,pval,expected",
        [
            (0.05, 0.05, 0.0),
            (0.05, 0.005, 1.0),
            (0.05, 0.0005, 2.0),
            (0.05, 0.5, -1.0),
            (0.01, 0.01, 0.0),
            (0.20, 0.20, 0.0),
            (0.025, 0.05, -0.3010299957),
            (0.05, 0.0, 310.0),
        ],
    )
    def test_wilson_significance(self, alpha, pval, expected):
        actual = core.wilson_significance(pval, alpha)
        assert actual == pytest.approx(expected)


class TestMaximumLikelihoodEstimation:
    def test_null_lift_zero(self):
        trials = [1000, 1000]
        successes = [100, 120]

        expected = [0.11, 0.11]
        actual = core.mle_under_null(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    def test_relative_lift(self):
        trials = [1000, 1000]
        successes = [100, 120]
        null_lift = 0.01

        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency
        #
        # p = cp.Variable(2)
        # objective = cp.Maximize(
        #     successes[0] * cp.log(p[0])
        #     + (trials[0] - successes[0]) * cp.log(1 - p[0])
        #     + successes[1] * cp.log(p[1])
        #     + (trials[1] - successes[1]) * cp.log(1 - p[1])
        # )
        # constraints = [0 <= p, p <= 1, p[1] == p[0] * (1 + null_lift)]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected = p.value
        expected = [0.10945852024217109, 0.1105531054445928]
        actual = core.mle_under_null(
            trials, successes, null_lift=null_lift, lift="relative"
        )

        assert actual[1] == pytest.approx(actual[0] * (1 + null_lift))
        assert actual == pytest.approx(expected)

    def test_absolute_lift(self):
        trials = [1000, 1000]
        successes = [100, 120]
        null_lift = 0.01

        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency
        #
        # p = cp.Variable(2)
        # objective = cp.Maximize(
        #     successes[0] * cp.log(p[0])
        #     + (trials[0] - successes[0]) * cp.log(1 - p[0])
        #     + successes[1] * cp.log(p[1])
        #     + (trials[1] - successes[1]) * cp.log(1 - p[1])
        # )
        # constraints = [0 <= p, p <= 1, p[1] == p[0] + null_lift]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected = p.value
        expected = [0.10480017, 0.11480017]
        actual = core.mle_under_null(
            trials, successes, null_lift=null_lift, lift="absolute"
        )

        assert actual[1] == pytest.approx(actual[0] + null_lift)
        assert actual == pytest.approx(expected, abs=1e-6)


class TestScoreTest:
    def test_null_lift(self):
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.4657435879336349
        actual = core.score_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    def test_null_lift_observed_relative(self):
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.10
        expected = 1.0
        actual = core.score_test(
            trials, successes, null_lift=null_lift, lift="relative"
        )
        assert actual == pytest.approx(expected)

    def test_null_lift_observed_absolute(self):
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.01
        expected = 1.0
        actual = core.score_test(
            trials, successes, null_lift=null_lift, lift="absolute"
        )
        assert actual == pytest.approx(expected)

    def test_null_lift_observed(self):
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.10
        expected = 1.0
        actual = core.score_test(
            trials, successes, null_lift=null_lift, lift="relative"
        )
        assert actual == pytest.approx(expected)

    def test_no_difference(self):
        trials = [1000, 1000]
        successes = [100, 100]
        expected = 1.0
        actual = core.score_test(trials, successes, null_lift=0.0)
        assert actual == expected

    def test_symmetric(self):
        trials = [1000, 1000]
        successes = [100, 110]
        one = core.score_test(trials, successes, null_lift=0.0)
        two = core.score_test(
            list(reversed(trials)), list(reversed(successes)), null_lift=0.0
        )
        assert one == two

    def test_extremes_01(self):
        trials = [1000, 1000]
        successes = [0, 1]
        expected = 0.3171894922467479
        actual = core.score_test(trials, successes)
        assert actual == pytest.approx(expected)

    def test_extremes_00(self):
        trials = [1000, 1000]
        successes = [0, 0]
        expected = 1
        actual = core.score_test(trials, successes)
        assert actual == pytest.approx(expected)

    def test_extremes_22(self):
        trials = [2, 2]
        successes = [1, 2]
        expected = 0.2482130789
        actual = core.score_test(trials, successes)
        assert actual == pytest.approx(expected)

    # @pytest.mark.slow
    def test_coverage(self, capsys):
        # Takes about 2 minutes to run on my machine
        p0 = 0.01
        N = 1000  # In each group
        trials = np.array([N, N])
        B = 1_000_000
        alpha = 0.05
        crit = ss.chi2.isf(alpha, df=1)
        z = ss.norm.isf(alpha / 2)
        wdth = alpha * (1 - alpha) / B + z * z / (4 * B * B)
        wdth = z / (1 + z * z / B) * math.sqrt(wdth)

        bs = 0
        bl = 0
        bz = 0
        np.random.seed(1)
        successes = np.random.binomial(N, p0, size=(B, 2))
        for i in range(B):
            if core.score_test(trials, successes[i, :], crit=crit):
                bs += 1

            if core.likelihood_ratio_test(trials, successes[i, :], crit=crit):
                bl += 1

            if core.z_test(trials, successes[i, :], crit=z):
                bz += 1

        lbs, ubs = core.wilson_interval(bs, B)
        lbl, ubl = core.wilson_interval(bl, B)
        lbz, ubz = core.wilson_interval(bz, B)
        with capsys.disabled():
            print(f"With {B:,d} trials, half width of 95% conf int: {wdth:.04%}")
            print(f"Score test: b={bs} => {bs/B:.04%} in ({lbs:.04%}, {ubs:.04%})")
            print(f"LR test: b={bl} => {bl/B:.04%} in ({lbl:.04%}, {ubl:.04%})")
            print(f"Z test: b={bz} => {bz/B:.04%} in ({lbz:.04%}, {ubz:.04%})")

        tol = 0.0028
        assert lbs - tol <= alpha
        assert lbl - tol <= alpha
        assert lbz - tol <= alpha
        assert alpha <= ubs + tol
        assert alpha <= ubl + tol
        assert alpha <= ubz + tol


class TestLikelihoodRatioTest:
    def test_null_lift(self):
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Pretty close!
        expected = 0.4656679698948981
        actual = core.likelihood_ratio_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    def test_symmetric(self):
        trials = [1000, 1000]
        successes = [100, 110]
        one = core.likelihood_ratio_test(trials, successes, null_lift=0.0)
        two = core.likelihood_ratio_test(
            list(reversed(trials)), list(reversed(successes)), null_lift=0.0
        )
        assert one == two


class TestZTest:
    def test_null_lift(self):
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Pretty close!
        expected = 0.46574358793363524
        actual = core.z_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    def test_symmetric(self):
        trials = [1000, 1000]
        successes = [100, 110]
        one = core.z_test(trials, successes, null_lift=0.0)
        two = core.z_test(
            list(reversed(trials)), list(reversed(successes)), null_lift=0.0
        )
        assert one == two


class TestConfidenceInterval:
    def test_conf_int_relative(self):
        trials = [1000, 1000]
        successes = [100, 110]
        expected_low = -0.14798553466796882
        expected_high = 0.4204476928710939

        actual_low, actual_high = core.confidence_interval(
            trials,
            successes,
            alpha=0.05,
            lift="relative",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    def test_conf_int_absolute(self):
        trials = [1000, 1000]
        successes = [100, 110]
        expected_low = -0.016966857910156258
        expected_high = 0.037053527832031245

        actual_low, actual_high = core.confidence_interval(
            trials,
            successes,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    def test_extremes_10(self):
        trials = [1000, 1000]
        successes = [1, 0]
        expected_low = -1.0
        expected_high = 2.8384127807617183
        actual_low, actual_high = core.confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == pytest.approx(expected_high)

    def test_extremes_01(self):
        trials = [1000, 1000]
        successes = [0, 1]
        expected_low = "-Infinity"
        expected_high = "Infinity"
        actual_low, actual_high = core.confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == expected_high

    def test_extremes_00(self):
        trials = [1000, 1000]
        successes = [0, 0]
        expected_low = "-Infinity"
        expected_high = "Infinity"
        actual_low, actual_high = core.confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == expected_high

    def test_conf_int_relative_lrt(self):
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.14798553466796882 for score test
        expected_low = -0.14850128173828125
        # Compare:      0.4204476928710939 for score test
        expected_high = 0.4228744506835937

        actual_low, actual_high = core.confidence_interval(
            trials,
            successes,
            test=core.likelihood_ratio_test,
            alpha=0.05,
            lift="relative",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    def test_conf_int_absolute_lrt(self):
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016909484863281254
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036967468261718754

        actual_low, actual_high = core.confidence_interval(
            trials,
            successes,
            test=core.likelihood_ratio_test,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    def test_conf_int_absolute_z(self):
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016966857910156258
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.037053527832031245

        actual_low, actual_high = core.confidence_interval(
            trials,
            successes,
            test=core.z_test,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)


class TestScorePower:
    def test_power(self):
        trials = [1000, 1000]
        p_null = [0.1, 0.1]
        p_alt = [0.07692307692307691, 0.11538461538461536]
        expected = 0.8323679253014326

        actual = core.score_power(trials, p_null, p_alt, alpha=0.05)
        assert actual == pytest.approx(expected)

    def test_abtest_power_relative_lift(self):
        baseline = 0.10
        alt_lift = 0.50
        group_sizes = [1000, 1000]
        expected = 0.8323679253014326

        actual = core.abtest_power(group_sizes, baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    def test_abtest_power_absolute_lift(self):
        baseline = 0.10
        alt_lift = 0.04
        group_sizes = [1000, 1000]
        expected = 0.8464821088914328

        actual = core.abtest_power(group_sizes, baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    def test_minimum_detectable_lift_relative_lift(self):
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.47324371337890625

        actual = core.minimum_detectable_lift(group_sizes, baseline, lift="relative")
        assert actual == pytest.approx(expected)

    def test_minimum_detectable_lift_absolute_lift(self):
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.03758753299713134

        actual = core.minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        assert actual == pytest.approx(expected)

    def test_minimum_detectable_drop(self):
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.32122573852539066

        actual = core.minimum_detectable_lift(group_sizes, baseline, drop=True)
        assert actual == pytest.approx(expected)

    def test_required_sample_size_relative_lift(self):
        baseline = 0.10
        alt_lift = 0.50
        expected = 1843

        actual = core.required_sample_size(baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    def test_required_sample_size_absolute_lift(self):
        baseline = 0.10
        alt_lift = 0.05
        expected = 1132

        actual = core.required_sample_size(baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    # @pytest.mark.slow
    def test_coverage(self, capsys):
        # Takes about 20 seconds on my machine
        alpha = 0.05
        B = 1_000_000
        crit = ss.chi2.isf(alpha, df=1)

        trials = [10000, 10000]
        baseline = 0.01
        null_lift = 0.0
        alt_lift = 0.50

        p_null, p_alt = core.simple_hypothesis_from_composite(
            trials, baseline, null_lift, alt_lift
        )

        expected = core.score_power(trials, p_null, p_alt, alpha=alpha)

        b = 0
        np.random.seed(1)
        for i in range(B):
            successes = [np.random.binomial(ti, pi) for (ti, pi) in zip(trials, p_alt)]
            if core.score_test(trials, successes, crit=crit):
                b += 1

        lb, ub = core.wilson_interval(b, B)
        with capsys.disabled():
            print(f"Predicted power: {expected:.03%}")
            print(
                f"Rejected null {b}/{B} times => {b/B:0.3%} in ({lb:.03%}, {ub:.03%})"
            )

        tol = 0.0026
        assert lb - tol <= expected
        assert expected <= ub + tol
