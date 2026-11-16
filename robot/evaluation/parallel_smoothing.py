"""Offline smoothness comparison; independent root truth never enters control."""

import math

import numpy as np


def _longest_interval(times, mask):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    groups = [g for g in groups if len(g)]
    if not groups:
        return dict(duration_seconds=0., start_source_seconds=None, end_source_seconds=None)
    group = max(groups, key=lambda g: times[g[-1]] - times[g[0]])
    return dict(duration_seconds=float(times[group[-1]] - times[group[0]]),
        start_source_seconds=float(times[group[0]]), end_source_seconds=float(times[group[-1]]))


def summarize_parallel_smoothness(report, truth):
    """Measure translation, including nonzero-command creeping and in-place turns.

    Distinct ordered samples must cover the walking encounter. Finite-difference
    root speed is an oracle evaluation quantity, not an estimator input. Existing
    native safety/arrival/curve acceptance remains mandatory and unchanged.
    """
    if len(truth) < 30:
        raise ValueError('smoothness requires at least30 ordered independent samples')
    t = np.array([r['time'] for r in truth], float)
    robot = np.array([r['robot_xy'] for r in truth], float)
    person = np.array([r['actor_xyz'][:2] for r in truth], float)
    if (robot.shape != (len(t), 2) or person.shape != robot.shape
        or not all(np.isfinite(a).all() for a in (t, robot, person))
        or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .15 + 1e-9):
        raise ValueError('smoothness requires finite ordered poses with source gaps <=.15s')
    robot_speed = np.linalg.norm(np.gradient(robot, t, axis=0), axis=1)
    person_speed = np.linalg.norm(np.gradient(person, t, axis=0), axis=1)
    near = ((np.array([r['phase'] for r in truth]) == 'move') & (person_speed >= .3)
        & (np.linalg.norm(robot - person, axis=1) <= 2.))
    curve = report.get('human_navigation_validation', {}).get('parallel_curve', {})
    checks = dict(native_navigation_passed=report.get('passed') is True,
        native_parallel_curve_passed=curve.get('passed') is True,
        explicit_parallel_scope=report.get('native_human_mode') == 'parallel',
        fixed_lambda=report.get('fixed_lambda') == 1.25,
        enough_nearby_walking_samples=int(near.sum()) >= 30)
    return dict(eligible=all(checks.values()), checks=checks,
        nearby_walking_samples=int(near.sum()),
        longest_translation_hold=_longest_interval(t, near & (robot_speed < .02)),
        longest_slow_translation=_longest_interval(t, near & (robot_speed < .05)),
        hold_speed_threshold_mps=.02, slow_speed_threshold_mps=.05,
        minimum_clearance_m=report.get('minimum_clearance_m'),
        navigation_simulation_seconds=report.get('navigation_simulation_seconds'),
        scope='post-run actual root translation during walking within2m; commanded motion/rotation cannot hide a hold; existing native acceptance unchanged; no real-time or repeatability claim')


def activation_only_command_change(baseline_command, candidate_command):
    """Require the same invocation except output path and one larger activation."""
    values, commands = [], []
    for command in (baseline_command, candidate_command):
        command = list(command)
        if command.count('--output') != 1 or command.count('--guidance-activation-h') != 1:
            raise ValueError('one explicit output and activation argument required')
        try:
            i = command.index('--guidance-activation-h')
            value = float(command[i+1])
            command[i+1] = '<activation>'
            command[command.index('--output')+1] = '<output>'
        except (IndexError, ValueError) as error:
            raise ValueError('complete output/activation arguments required') from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError('finite positive field activation required')
        values.append(value)
        commands.append(command)
    return dict(matched=commands[0] == commands[1] and values[1] > values[0],
        baseline_activation_h=values[0], candidate_activation_h=values[1],
        scope='field activation values, not metres; all remaining invocation arguments identical')


def outward_only_command_change(baseline_command, candidate_command):
    """Require one new reference preference, never a changed physical argument."""
    flag = '--guidance-outward-reference-fraction'
    values, commands = [], []
    for command in (baseline_command, candidate_command):
        command = list(command)
        if command.count('--output') != 1 or command.count(flag) > 1:
            raise ValueError('one explicit output and at most one outward preference required')
        try:
            command[command.index('--output')+1] = '<output>'
            value = 0.
            if flag in command:
                i = command.index(flag)
                value = float(command[i+1])
                del command[i:i+2]
        except (IndexError, ValueError) as error:
            raise ValueError('complete output/outward arguments required') from error
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('outward reference fraction must be in[0,1]')
        values.append(value)
        commands.append(command)
    return dict(matched=commands[0] == commands[1] and values[1] > values[0],
        baseline_outward_fraction=values[0], candidate_outward_fraction=values[1],
        scope='normalized current-observed travel reference only; all remaining invocation arguments identical')


def outward_identity_command_change(baseline_command, candidate_command):
    """Audit the separately declared reference AND short identity-policy variant."""
    flag = '--human-passing-identity-prior-max-age'
    ages, commands = [], []
    for command in (baseline_command, candidate_command):
        command = list(command)
        if command.count(flag) != 1:
            raise ValueError('one explicit identity-selection age required')
        try:
            i = command.index(flag)
            age = float(command[i+1])
            del command[i:i+2]
        except (IndexError, ValueError) as error:
            raise ValueError('complete identity-selection argument required') from error
        if not math.isfinite(age) or not 0 <= age <= .5:
            raise ValueError('identity-selection age must be in[0,.5]')
        ages.append(age)
        commands.append(command)
    result = outward_only_command_change(*commands)
    result.update(matched=result['matched'] and ages[0] == 0 and ages[1] > 0,
        baseline_identity_prior_age_seconds=ages[0], candidate_identity_prior_age_seconds=ages[1],
        scope='separate outward reference plus bounded identity-selection PRIOR; every other invocation argument identical; excluded from primary comparison')
    return result


def translation_only_command_change(baseline_command, candidate_command):
    """Audit one turning-reference option with unchanged identity/geometry policy."""
    flag = '--guidance-turn-translation-fraction'
    values, commands = [], []
    for command in (baseline_command, candidate_command):
        command = list(command)
        if command.count('--output') != 1 or command.count(flag) > 1:
            raise ValueError('one output and at most one turn-translation preference required')
        try:
            command[command.index('--output')+1] = '<output>'
            value = 0.
            if flag in command:
                i = command.index(flag)
                value = float(command[i+1])
                del command[i:i+2]
        except (IndexError, ValueError) as error:
            raise ValueError('complete output/translation arguments required') from error
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('turn-translation preference must be in[0,1]')
        values.append(value)
        commands.append(command)
    return dict(matched=commands[0] == commands[1] and values[0] == 0 and values[1] > 0,
        baseline_turn_translation_fraction=values[0], candidate_turn_translation_fraction=values[1],
        scope='turning-reference preference only before original affine/exact safety checks; all remaining invocation arguments identical')


def compare_parallel_smoothness(baseline, candidate, *, matched_configuration,
                               maximum_hold_seconds=.5, minimum_reduction_fraction=.5):
    if (not math.isfinite(maximum_hold_seconds) or maximum_hold_seconds < 0
        or not math.isfinite(minimum_reduction_fraction) or not 0 <= minimum_reduction_fraction <= 1):
        raise ValueError('finite nonnegative hold target and reduction fraction in[0,1] required')
    old = baseline['longest_translation_hold']['duration_seconds']
    new = candidate['longest_translation_hold']['duration_seconds']
    reduction = None if old == 0 else (old-new)/old
    checks = dict(baseline_native_acceptance=baseline['eligible'],
        candidate_native_acceptance=candidate['eligible'], matched_configuration=bool(matched_configuration),
        bounded_actual_translation_hold=new <= maximum_hold_seconds + 1e-9,
        hold_reduced=(new <= 1e-9 if old == 0 else reduction + 1e-9 >= minimum_reduction_fraction))
    return dict(passed=all(checks.values()), checks=checks, baseline=baseline, candidate=candidate,
        maximum_hold_seconds=maximum_hold_seconds, minimum_reduction_fraction=minimum_reduction_fraction,
        observed_hold_reduction_fraction=reduction,
        scope='separate matched controller-reference engineering comparison; original safety/arrival/curve gates mandatory; no relaxed geometry, primary pooling, causal CRA or general reliability claim')
