"""Maximum-member field policy; individual four-factor assessments stay intact."""


def maximum_member_policy(member_policies):
    if len(member_policies) < 2:
        raise ValueError("a group needs at least two independently assessed members")
    members = sorted(member_policies)
    if any(member_policies[k].tangential_strength != 1.25 for k in members):
        raise ValueError("primary carried group policy requires fixed lambda 1.25")
    if len({member_policies[k].passing_side for k in members}) != 1:
        raise ValueError("group members require a consistent passing-side policy")
    winner = max(members, key=lambda k: member_policies[k].risk)
    result = member_policies[winner]
    return result, {
        "members": members,
        "member_phi": {str(k): member_policies[k].risk for k in members},
        "phi": result.risk,
        "dominant_member": winner,
        "basis": "maximum_member_phi",
        "scope": "shared field policy over member occupancy; no combined mass/factors or cumulative-risk claim",
    }
