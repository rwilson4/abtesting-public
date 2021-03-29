import math

import numpy as np
import scipy.stats as ss


def observed_lift(trials, successes, lift="relative"):
    pa = successes[0] / trials[0]
    pb = successes[1] / trials[1]
    if lift == "relative":
        ote = (pb - pa) / pa
    else:
        ote = pb - pa
    return ote


def mle_under_null(trials, successes, null_lift=0.0, lift="relative"):
    """Maximum Likelihood Estimation under H0

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes.

    Returns
    -------
     p : array
        Array [pa_star, pb_star], corresponding to the MLE of pa and pb under
        H0.

    Notes
    -----
    Solves the following optimization problem:
        maximize ll(pa, pb)
        s.t.     H0
    where H0 is of the form A*(p-a) = 0.

    When the null lift is zero, that corresponds to pa = pb, or A = [1 -1] and
    a = [0 0]'.

    Otherwise, the form of A and a depends on whether we are using relative
    lift (pb = pa * (1 + d)) or absolute lift (pb = pa + d). With relative
    lift, a = [0 0]' and A = [(1 + d) -1]. With absolute lift, a = [d 0]' and
    A = [1 -1].

    In all cases, H0 takes the form of a linear equality constraint. In all
    cases, the log-likelihood is concave, so the problem is efficiently
    solvable.

    When the null lift is zero, the solution is trivial. When we are using
    relative lift, there is still a fixed formula for the solution, but it is
    complicated! It involves solving a quadratic equation. When using absolute
    lift, we need to find the root of a cubic polynomial. It is easiest to use
    Newton's method for this.

    """
    if null_lift == 0:
        # MLE of parameters under null hypothesis
        p = [sum(successes) / sum(trials) for _ in range(2)]
    elif lift == "relative":
        S = sum(successes)
        T = sum(trials)
        neg_b = T + S + null_lift * (S + trials[1] - successes[1])
        a = T * (1 + null_lift)
        c = S
        radical = neg_b * neg_b - 4 * a * c
        pstar_a = (neg_b - math.sqrt(radical)) / (2.0 * a)
        pstar_b = pstar_a * (1.0 + null_lift)
        p = [pstar_a, pstar_b]
    else:
        # Find the root of the equation:
        #    A * x^3 + B * x^2 + C * x + D = 0
        val_tol = 1e-12
        step_tol = 1e-12

        sa = successes[0]
        sb = successes[1]
        fa = trials[0] - successes[0]
        fb = trials[1] - successes[1]
        d = null_lift

        A = sa + sb + fa + fb
        B = -2 * sa * (1 - d) + sb * (d - 2) - fa * (1 - 2 * d) - fb * (1 - d)
        C = sa * ((1 - d) ** 2 - d) + sb * (1 - d) - fa * d * (1 - d) - fb * d
        D = sa * d * (1 - d)

        pcrit = B * B - 3 * A * C
        if pcrit > 0:
            sqrt_pcrit = math.sqrt(pcrit)
            one_over_6A = 1.0 / (6 * A)
            pcrit_minus = (-B - sqrt_pcrit) * one_over_6A
            pcrit_plus = (-B + sqrt_pcrit) * one_over_6A

            if pcrit_minus < 0:
                pcrit_minus = 0.0

            if pcrit_plus > 1:
                pcrit_plus = 1.0
        else:
            pcrit_minus = 0
            pcrit_plus = 1

        x0 = 0.5 * (pcrit_minus + pcrit_plus)
        for _ in range(50):
            fn = A * x0 * x0 * x0 + B * x0 * x0 + C * x0 + D
            fpn = 3 * A * x0 * x0 + 2 * B * x0 + C
            x0 -= fn / fpn

            if abs(fn) < val_tol and abs(fn / fpn) < step_tol:
                break
        else:
            raise ValueError("MLE did not converge")

        pstar_a = x0
        pstar_b = pstar_a + null_lift
        p = [pstar_a, pstar_b]

    return p


def mle_under_alternative(trials, successes, alt_lift=None, lift="relative"):
    """Maximum Likelihood Estimation under H1

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     alt_lift : float, optional
        Lift associated with alternative hypothesis. If None (default),
        alternative is unconstrained.
     lift : ["relative", "absolute"], optional
        Whether to interpret `alt_lift` relative to the baseline success
        rate, or in absolute terms. See Notes.

    Returns
    -------
     p : array
        Array [pa_star, pb_star], corresponding to the MLE of pa and pb under
        H1.

    Notes
    -----
    The most common alternative hypothesis considered is unconstrained, in
    which case p is simply successes / trials. But we also support an
    alternative hypothesis of the same form as H0, in case we ever want that.

    """
    if alt_lift is None:
        successes = np.array(successes)
        trials = np.array(trials)
        return successes / trials
    else:
        return mle_under_null(trials, successes, null_lift=alt_lift, lift=lift)


def score_test(trials, successes, null_lift=0.0, lift="relative", crit=None):
    """Rao's score test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
     crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
     pval : float
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.

    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Not implemented")

    p = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)

    if min(p) <= 1e-12 or max(p) + 1e-12 >= 1.0:
        return 1.0

    # Score
    u = [(si - ni * pi) / (pi * (1 - pi)) for ni, si, pi in zip(trials, successes, p)]

    # Fisher information
    diagI = [ni / (pi * (1 - pi)) for ni, pi in zip(trials, p)]

    # Test statistic
    ts = sum([ui * ui / ii for ui, ii in zip(u, diagI)])

    if crit is None:
        # Note: this line takes 80% of the time for the score test, including the
        # MLE, which is really fast! So there's no real point in optimizing
        # anything else here. On the other hand, if we can optimize this line, then
        # great!
        pval = ss.chi2.sf(ts, df=1)
        return pval
    else:
        return ts >= crit


def likelihood_ratio_test(trials, successes, null_lift=0.0, lift="relative", crit=None):
    """Likelihood ratio test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.
     crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
     pval : float
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.

    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Not implemented")

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    if min(p0) <= 1e-12 or max(p0) + 1e-12 >= 1.0:
        return 1.0

    if min(p1) <= 1e-12 or max(p1) + 1e-12 >= 1.0:
        return 1.0

    def log_likelihood(p):
        return sum(
            [
                si * np.log(pi) + (ti - si) * np.log(1 - pi)
                for (si, ti, pi) in zip(successes, trials, p)
            ]
        )

    ts = 2 * (log_likelihood(p1) - log_likelihood(p0))
    if crit is None:
        pval = ss.chi2.sf(ts, df=1)
        return pval
    else:
        return ts >= crit


def z_test(trials, successes, null_lift=0.0, lift="relative", crit=None):
    """Z test for 2x2 contingency table.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     null_lift : float
        Lift associated with null hypothesis. Defaults to 0.0.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. Only absolute lift is currently supported,
        but a relative lift with null_lift 0 is also supported since this is
        equivalent to an absolute lift with null_lift 0. See Notes in
        `maximum_likelihood_estimation`.
     crit : float, optional
        Critical value for the test statistic. If omitted, a p-value will be
        returned. If passed, a boolean will be returned corresponding to
        whether the result is statistically significant. Useful primarily for
        simulations where we will be repeatedly assessing significance, since
        calculating the critical value can be done once instead of repeatedly.
        This makes such simulations about 5x faster.

    Returns
    -------
     pval : float
        P-value. Returned if `crit` is None.
     stat_sig : boolean
        True if the result is statistically significant, i.e. if the test
        statistic is >= `crit`. Returned if `crit` is not None.

    Notes
    -----
    Only supports two experiment groups at this time.

    """
    if len(trials) > 2 or len(successes) > 2:
        raise NotImplementedError("Not implemented")

    if lift == "relative" and null_lift != 0.0:
        raise NotImplementedError("Not Implemented")

    p0 = mle_under_null(trials, successes, null_lift=null_lift, lift=lift)
    p1 = mle_under_alternative(trials, successes)

    sigma2 = sum([pi * (1 - pi) / ti for (pi, ti) in zip(p0, trials)])
    z = (p1[1] - p1[0] - null_lift) / math.sqrt(sigma2)
    if crit is None:
        return 2.0 * ss.norm.cdf(-abs(z))
    else:
        return abs(z) >= crit


def confidence_interval(
    trials, successes, test=score_test, alpha=0.05, lift="relative"
):
    """Confidence interval for relative lift.

    Parameters
    ----------
     trials : array_like
        Number of trials in each group.
     successes : array_like
        Number of successes in each group.
     test : function
        A function implementing a significance test. Defaults to
        `score_test`. This function should in turn have arguments,
        trials, successes, null_lift, and return a p-value.
     alpha : float
        Threshold for significance. The confidence interval will have
        level 100(1-alpha)%. Defaults to 0.05, corresponding to a 95%
        confidence interval.
     lift : ["relative", "absolute"]
        Whether to interpret the null lift relative to the baseline success
        rate, or in absolute terms. See Notes in
        `maximum_likelihood_estimation`.

    Returns
    -------
     ci_low, ci_high : float
        Lower and upper bounds on a confidence interval.

    Notes
    -----
    Uses binary search to compute a confidence interval.

    """

    tol = 1e-6
    lower_bound_exists = True
    upper_bound_exists = True
    try:
        ote = observed_lift(trials, successes, lift=lift)
    except ZeroDivisionError:
        ote = 1.0
        upper_bound_exists = False

    if lift == "relative":
        lb_lb = ote - 0.01
        lb_ub = ote
        ub_lb = ote
        ub_ub = ote + 0.01
    else:
        pa = successes[0] / trials[0]
        lb_lb = max(ote - 0.01, -pa)
        lb_ub = ote
        ub_lb = ote
        ub_ub = min(ote + 0.01, 1.0 - pa)

    # Initial search for a lower bound on the lower bound of the
    # confidence interval.
    eps = 0.01
    lower_bound_exists = True
    while True:
        if (lift == "relative" and lb_lb < -1) or (lift == "absolute" and lb_lb < -pa):
            lower_bound_exists = False
            break

        pval = test(trials, successes, null_lift=lb_lb, lift=lift)
        if pval >= alpha:
            # lb_lb is consistent with data; decrease it
            lb_ub = lb_lb
            lb_lb -= eps
            eps *= 2
        else:
            break

    if lower_bound_exists:
        # (lb_lb, lb_ub) is a bound on the lower bound of the confidence interval
        while (lb_ub - lb_lb) > tol:
            lb = 0.5 * (lb_lb + lb_ub)
            pval = test(trials, successes, null_lift=lb, lift=lift)
            if pval >= alpha:
                # lb is consistent with data; expand the interval by
                # decreasing lb.
                lb_ub = lb
            else:
                # lb is rejected by the test; shrink the interval by
                # increasing lb.
                lb_lb = lb

        lb = 0.5 * (lb_lb + lb_ub)
    elif successes[0] > 0:
        lb = -1.0
    else:
        lb = "-Infinity"

    # Initial search for an upper bound on the upper bound of the
    # confidence interval.
    if upper_bound_exists:
        eps = 0.01
        while True:
            if ub_ub > 100:
                upper_bound_exists = False
                break

            pval = test(trials, successes, null_lift=ub_ub, lift=lift)
            if pval >= alpha:
                # ub_ub is consistent with data; increase it
                ub_lb = ub_ub
                ub_ub += eps
                eps *= 2
            else:
                break

    if upper_bound_exists:
        # (ub_lb, ub_ub) is a bound on the upper bound of the confidence interval
        while (ub_ub - ub_lb) > tol:
            ub = 0.5 * (ub_lb + ub_ub)
            pval = test(trials, successes, null_lift=ub, lift=lift)
            if pval >= alpha:
                # ub is consistent with data; expand the interval by
                # increasing ub.
                ub_lb = ub
            else:
                # ub is rejected by the test; shrink the interval by
                # decreasing ub.
                ub_ub = ub

        ub = 0.5 * (ub_lb + ub_ub)
    elif lift == "relative":
        ub = "Infinity"
    else:
        ub = 1.0

    return (lb, ub)


def score_power(n, p_null, p_alt, alpha=0.05):
    """Power of Rao's Score Test

    Parameters
    ----------
     n : array_like
        Number of experimental units in each group.
     p_null : array_like
        Probability of success in each group under the null
        hypothesis.
     p_alt : array_like
        Probability of success in each group under the alternative
        hypothesis.
     alpha : float
        Type-I error rate. Defaults to 0.05

    Returns
    -------
     power : float
        The power of the test.

    Notes
    -----
    Rao's score test is the same as Pearson's chi-squared test for 2x2
    contingency tables, so the power has a nice simple form.

    """
    nc = 0.0
    for (ni, null, alt) in zip(n, p_null, p_alt):
        nc += ni * (null - alt) * (null - alt) / (null * (1.0 - null))

    return ss.ncx2.sf(ss.chi2.isf(alpha, df=1), df=1, nc=nc)


def abtest_power(
    group_sizes,
    baseline,
    alt_lift,
    alpha=0.05,
    null_lift=0.0,
    power=score_power,
    lift="relative",
):
    """Power associated with an A/B Test

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Lift associated with alternative hypothesis.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     power : float
        The power of the test.

    """
    if len(group_sizes) > 2:
        # Get two smallest groups -- this governs the overall power
        a, b, *_ = np.partition(group_sizes, 1)
        group_sizes = [a, b]

    p_null, p_alt = simple_hypothesis_from_composite(
        group_sizes, baseline, null_lift, alt_lift, lift=lift
    )
    return power(group_sizes, p_null, p_alt, alpha=alpha)


def simple_hypothesis_from_composite(
    group_sizes, baseline, null_lift, alt_lift, lift="relative"
):
    """Translate a composite hypothesis into a simple hypothesis.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     null_lift : float
        Lift associated with null hypothesis.
     alt_lift : float
        Lift associated with alternative hypothesis.
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     p_null, p_alt: array
        Success rate in each group under the null and alternative hypotheses,
        respectively.

    Notes
    -----
    The power formula relies on simple null and alternative hypotheses of the
    form: "under the null hypothesis, the success rate in the first group is x,
    and the success rate in the second group is y". We care more about
    specifying composite hypotheses like, "under the null hypothesis, the
    success rates in the two groups are equal", or "under the alternative
    hypothesis, the success rate in the second group is 10% higher than in the
    first group".

    This function translates such composite null and alternative hypotheses to
    simple hypotheses. We need the baseline success rate as well. Consider this
    translation mechanism:
       H0: pa = baseline, pb = baseline * (1 + null_lift)
       H1: pa = baseline, pb = baseline * (1 + alt_lift)
    The success rate in the first group is the same either way, but the success
    rate in the second group depends on the null/alt lift. This seems innocent
    enough, but consider the noncentrality parameter of the chi-2 distribution
    under the alternative hypothesis:

                 na * (pa - pi_a)^2     nb * (pb - pi_b)^2
      lambda =   ------------------  +  ------------------ ,
                  pi_a * (1 - pi_a)      pi_b * (1 - pi_b)

    where pi_a, pi_b are the success rates in the first and second groups under
    H0, pa and pb are for H1, and na and nb are the group sizes. In our naive
    approach, pa is always equal to pi_a (is equal to the baseline), so the
    first term is always zero. No matter how big or small na is, the
    noncentrality parameter is always the same, so the power is always the
    same. That's not right.

    The power of the test increases with lambda, so we might reasonably ask,
    what is the lowest lambda could be while still being aligned with the
    information given? If we leave the success rates under H0 alone (pi_a =
    baseline and pi_b = pi_a * (1 + null_lift)), which seems reasonable enough,
    we can minimize lambda subject to the constraint pb = pa * (1 + alt_lift).
    Treating na, nb, pi_a, and pi_b as data, this is a convex optimization
    problem. The solution (pa, pb) is the simple alternative hypothesis.

    """
    na = group_sizes[0]
    nb = group_sizes[1]

    p_null_a = baseline
    if lift == "relative":
        p_null_b = (1 + null_lift) * baseline
    else:
        p_null_b = baseline + null_lift

    if lift == "relative":
        p_alt_a = (2 * na / (1 - p_null_a)) + (2 * nb * (1 + alt_lift) / (1 - p_null_b))
        p_alt_a /= (2 * na) / (p_null_a * (1 - p_null_a)) + (
            2 * nb * (1 + alt_lift) ** 2
        ) / (p_null_b * (1 - p_null_b))
    else:
        p_alt_a = 2 * na / (1 - p_null_a)
        p_alt_a += 2 * nb * (p_null_b - alt_lift) / (p_null_b * (1 - p_null_b))
        p_alt_a /= 2 * na / (p_null_a * (1 - p_null_a)) + 2 * nb / (
            p_null_b * (1 - p_null_b)
        )

    if lift == "relative":
        p_alt_b = (1 + alt_lift) * p_alt_a
    else:
        p_alt_b = p_alt_a + alt_lift

    p_null = [p_null_a, p_null_b]
    p_alt = [p_alt_a, p_alt_b]
    return p_null, p_alt


def minimum_detectable_lift(
    group_sizes,
    baseline,
    alpha=0.05,
    beta=0.2,
    null_lift=0.0,
    power=score_power,
    drop=False,
    lift="relative",
):
    """Minimum detectable lift.

    Parameters
    ----------
     group_sizes : array_like
        Number of experimental units in each group.
     baseline : float
        Baseline success rate associated with first experiment group.
     alpha : float, optional
        Type-I error rate threshold. Defaults to 0.05.
     beta : float, optional
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     null_lift : float, optional
        Lift associated with null hypothesis. Defaults to 0.0.
     power : function, optional
        Function that computes power, such as `score_power` (default).
     drop : boolean, optional
        If True, the minimum detectable drop will be returned.
        Defaults to False, returning the minimum detectable lift.
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     mdl : float
        Minimum detectable lift/drop associated with test. If `lift` is
        "relative", this will be in relative terms, otherwise it will be in
        absolute terms.

    Notes
    -----
    Uses binary search to compute the smallest lift/drop with adequate
    power.

    """

    tol = 1e-6

    # Find an extremum bound on the MDL
    mdl_inner = 0.0
    if drop:
        if lift == "relative":
            mdl_extremum = -0.2
        else:
            mdl_extremum = -0.99 * baseline
    else:
        if lift == "relative":
            mdl_extremum = 0.2
        else:
            mdl_extremum = 0.99 * (1 - baseline)

    pwr = abtest_power(
        group_sizes,
        baseline,
        mdl_extremum,
        alpha=alpha,
        null_lift=null_lift,
        power=power,
        lift=lift,
    )

    while pwr < 1 - beta:
        mdl_inner = mdl_extremum
        if lift == "relative":
            mdl_extremum *= 2
        else:
            mdl_extremum = 0.5 * (-baseline + mdl_extremum)

        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl_extremum,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )

    while abs(mdl_extremum - mdl_inner) > tol:
        mdl = 0.5 * (mdl_inner + mdl_extremum)
        pwr = abtest_power(
            group_sizes,
            baseline,
            mdl,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase mdl
            mdl_inner = mdl
        else:
            # Adequate power, decrease mdl
            mdl_extremum = mdl

    if drop:
        mdl_extremum *= -1.0

    return mdl_extremum


def required_sample_size(
    baseline,
    alt_lift,
    alpha=0.05,
    beta=0.2,
    group_proportions=None,
    null_lift=0.0,
    power=score_power,
    lift="relative",
):
    """Required sample size.

    Parameters
    ----------
     baseline : float
        Baseline success rate associated with first experiment group.
     alt_lift : float
        Relative lift (second group relative to first) associated with
        alternative hypothesis.
     alpha : float
        Type-I error rate threshold. Defaults to 0.05.
     beta : float
        Type-II error rate threshold (1 - power). Defaults to 0.2, or
        80% power.
     group_proportions : array_like or None
        Fraction of experimental units in each group. If None
        (default), will use an even split.
     null_lift : float
        Relative lift (second group relative to first) associated with
        null hypothesis. Defaults to 0.0.
     power : function
        Function that computes power, such as `score_power` (default).
     lift : ["relative", "absolute"], optional
        Whether to interpret the null/alternative lift relative to the baseline
        success rate, or in absolute terms. Defaults to "relative".

    Returns
    -------
     sample_size : int
        Minimum sample size, across all experiment groups, required to
        have desired sensitivity.

    Notes
    -----
    Uses binary search to compute the smallest sample size with
    adequate power.

    """

    tol = 0.01

    if group_proportions is None:
        group_proportions = [0.5, 0.5]

    def sample_size_to_group_sizes(ss):
        return [int(ss * g) for g in group_proportions]

    # Find an upper bound on the required sample size
    ss_lower = 0
    ss_upper = 1000

    pwr = abtest_power(
        sample_size_to_group_sizes(ss_upper),
        baseline,
        alt_lift,
        alpha=alpha,
        null_lift=null_lift,
        power=power,
        lift=lift,
    )

    while pwr < 1 - beta:
        ss_lower = ss_upper
        ss_upper *= 2
        pwr = abtest_power(
            sample_size_to_group_sizes(ss_upper),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )

    while ss_upper - ss_lower > tol * ss_lower:
        ss = int(0.5 * (ss_lower + ss_upper))
        pwr = abtest_power(
            sample_size_to_group_sizes(ss),
            baseline,
            alt_lift,
            alpha=alpha,
            null_lift=null_lift,
            power=power,
            lift=lift,
        )
        if pwr < 1 - beta:
            # Inadequate power, increase ss
            ss_lower = ss
        else:
            # Adequate power, decrease ss
            ss_upper = ss

    return ss_upper


def wilson_interval(s, n, alpha=0.05):
    """Wilson Confidence Interval on binomial proportion

    Parameters
    ----------
     s, n : int
        The number of successes, trials.
     alpha : float, optional
        The significance level. Defaults to 0.05, corresponding to a 95%
        confidence interval.

    Returns
    -------
     lb, ub : float
        Lower and upper bounds of a 100(1-`alpha`)% confidence interval on the
        binomial proportion.

    Notes
    -----
    Assuming s ~ Binom(n, p), this function returns a confidence interval on p.

    """
    z = ss.norm.isf(alpha / 2)
    ctr = (s + 0.5 * z * z) / (n + z * z)
    wdth = s * (n - s) / n + z * z / 4
    wdth = (z / (n + z * z)) * math.sqrt(wdth)
    return ctr - wdth, ctr + wdth


def wilson_significance(pval, alpha):
    """Wilson significance

    Parameters
    ----------
     pval : float
        P-value.
     alpha : float
        Type-I error threshold.

    Returns
    -------
     W : float
        Wilson significance: log10(alpha / pval).

    Notes
    -----
    The Wilson significance is defined to be log10(alpha / pval),
    where pval is the p-value and alpha is the Type-I error rate. It
    has the following properties:
     - When the result is statistically significant, W > 0.
     - The larger W, the stronger the evidence.
     - An increase in W of 1 corresponds to a 10x decrease in p-value.

    """
    try:
        W = math.log10(alpha) - math.log10(pval)
    except ValueError:
        W = 310.0

    return W
