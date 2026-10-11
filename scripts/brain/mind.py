"""Brain readout: template sentences from the brain's own state, next to the simulator's truth (no language model).
Identical templates in app/sim/brain/mind.js. Batch rows are handled one at a time (strings).
"""
import math

SECTORS = ["ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right"]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sector(rel):
    """Body-relative bearing (rad, + = left) -> one of 8 words."""
    return SECTORS[int(math.floor((wrap(rel) + math.pi / 8) % (2 * math.pi) / (math.pi / 4))) % 8]


def steering_slot(heading_cmd, bearings, strengths, visible):
    """The visible object (strength > 0.2) whose bearing is closest to the steering command, within 30 deg; else -1."""
    best, bd = -1, math.radians(30)
    for k, (b, s, v) in enumerate(zip(bearings, strengths, visible)):
        d = abs(wrap(b - heading_cmd))
        if v and s > 0.2 and d < bd:
            best, bd = k, d
    return best


def say(phase, heading_est, home_est, familiarity, slot, heading_true, home_true, roles, obj_dist, faults):
    """-> (text, truth). home_* are 2-vectors from drone to home (world frame for truth, compass frame for est)."""
    def home_words(v, hd):
        d = math.hypot(v[0], v[1])
        return f"{d:.0f} m {sector(math.atan2(v[1], v[0]) - hd)}" if d >= 1 else "here"
    text = f"home is {home_words(home_est, heading_est)}; target familiarity {familiarity:.2f}"
    if phase == "home":
        text += "; delivered, flying home"
    elif slot >= 0:
        text += f"; committed to object {slot}"
    else:
        text += "; searching"
    truth = f"truth: home {home_words(home_true, heading_true)}"
    if slot >= 0 and phase != "home":
        truth += f"; object {slot} is {'the target' if roles[slot] == 'target' else 'a look-alike'}, {obj_dist[slot]:.0f} m away"
    if faults:
        truth += "; faults: " + ", ".join(faults)
    return text, truth


if __name__ == "__main__":
    assert sector(0) == "ahead" and sector(math.pi / 2) == "left" and sector(-math.pi / 2) == "right"
    assert sector(math.pi) == "behind" and sector(math.radians(140)) == "behind-left"
    t, tr = say("search", 0.0, [-25.0, 3.0], 0.62, 2, 0.1, [-27.0, 5.0], ["decoy", "target", "decoy"], [10, 5, 6.2], ["false_sun"])
    assert t == "home is 25 m behind; target familiarity 0.62; committed to object 2", t
    assert tr == "truth: home 27 m behind; object 2 is a look-alike, 6 m away; faults: false_sun", tr
    assert steering_slot(0.1, [0.0, 1.0], [0.5, 0.9], [True, True]) == 0
    assert steering_slot(0.1, [0.0, 1.0], [0.1, 0.9], [True, True]) == -1
    print("mind self-check OK:", t, "|", tr)
