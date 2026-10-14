#!/usr/bin/env python3
"""Render "Which muscles work on a hike?" as an MP4.

A side-view hiker walks across flat ground, uphill, downhill and rocky
trail. The muscles doing the work glow (slate -> amber -> red) and pulse
with each step, with callout labels explaining what each one does.

Needs Python 3 with Pillow + numpy, and ffmpeg (libx264) on PATH.

    python3 scripts/hiking-video/render.py            # full video + poster
    python3 scripts/hiking-video/render.py --frames 2 6.5 14  # preview PNGs

The rendered video has no sound; run narrate.py afterwards to add the voice.
"""
import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_VIDEO = ROOT / "assets" / "videos" / "hiking-muscles.mp4"
OUT_POSTER = ROOT / "assets" / "images" / "hiking-muscles.png"

W, H, FPS = 1280, 720, 30
SS = 2  # supersampling factor: draw at 2x, downsample for smooth edges
GLOW_DIV = 4  # glow layer is rendered at 1/4 of the final resolution

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        _font_cache[key] = ImageFont.truetype(FONT_DIR + name, size * SS)
    return _font_cache[key]


# ---------------------------------------------------------------- timeline --

MUSCLES = ["calves", "quads", "hamstrings", "glutes", "hip_flexors",
           "shins", "core", "lower_back", "shoulders", "arms"]

# Labels: (muscle, title, subtitle, side, y). Side "L" boxes sit left of the
# hiker (muscles on the back of the body), "R" boxes sit right (front).
SCENES = [
    dict(key="title", heading=None, dur=3.5,
         slope=0.0, stride=100, period=1.1, lean=5, poles=0,
         muscles={}, labels=[], caption=None),
    dict(key="flat", heading="Flat trail", dur=6.5,
         slope=0.0, stride=100, period=1.1, lean=5, poles=0,
         muscles=dict(calves=.65, quads=.6, hamstrings=.55, glutes=.55,
                      hip_flexors=.25, shins=.25, core=.2),
         labels=[("glutes", "Glutes", "Swing the leg back", "L", 215),
                 ("hamstrings", "Hamstrings", "Pull the leg through", "L", 335),
                 ("calves", "Calves", "Push off the toes", "L", 455),
                 ("quads", "Quadriceps", "Straighten the knee", "R", 300)],
         caption="On flat ground every step uses calves, quads, hamstrings & glutes"),
    dict(key="uphill", heading="Uphill", dur=8.0,
         slope=-0.2, stride=92, period=1.3, lean=16, poles=0,
         muscles=dict(glutes=1.0, quads=.95, calves=.9, hip_flexors=.8,
                      hamstrings=.55, core=.35, lower_back=.3),
         labels=[("glutes", "Glutes", "Main engine - drive you up", "L", 210),
                 ("calves", "Calves", "Push off on every step", "L", 440),
                 ("hip_flexors", "Hip flexors", "Lift the knee to the next step", "R", 150),
                 ("quads", "Quadriceps", "Straighten the knee to lift you", "R", 265)],
         caption="Uphill: glutes and quads do the heavy lifting"),
    dict(key="downhill", heading="Downhill", dur=8.0,
         slope=0.2, stride=84, period=1.15, lean=-3, poles=0,
         muscles=dict(quads=1.0, shins=.9, hamstrings=.6, calves=.45,
                      glutes=.4, core=.45),
         labels=[("hamstrings", "Hamstrings", "Stabilize the knee", "L", 300),
                 ("calves", "Calves", "Soften each landing", "L", 420),
                 ("quads", "Quadriceps", "Brake every step (eccentric)", "R", 240),
                 ("shins", "Shins (tibialis)", "Control how the foot lands", "R", 380)],
         caption="Downhill: quads work as brakes - that's why they ache the next day"),
    dict(key="core", heading="Core & upper body", dur=7.5,
         slope=0.0, stride=100, period=1.15, lean=7, poles=1,
         muscles=dict(core=1.0, lower_back=.9, shoulders=.9, arms=.85,
                      glutes=.3, quads=.35, calves=.35, hamstrings=.3),
         labels=[("shoulders", "Shoulders & traps", "Carry the backpack straps", "L", 150),
                 ("lower_back", "Lower back", "Supports the pack's weight", "L", 300),
                 ("core", "Abs & obliques", "Keep you balanced on rocks", "R", 300),
                 ("arms", "Arms & triceps", "Push on the trekking poles", "R", 175)],
         caption="Core & upper body: balance, backpack and trekking poles"),
    dict(key="recap", heading="Recap", dur=8.0,
         slope=0.0, stride=100, period=1.1, lean=5, poles=0,
         muscles={m: .85 for m in MUSCLES},
         labels=[], caption="Tip: stretch your calves, quads and hips after the hike"),
]

_t = 0.0
for _s in SCENES:
    _s["start"] = _t
    _t += _s["dur"]
TOTAL = _t

RECAP_ROWS = [
    ("Glutes", "climbing"),
    ("Quadriceps", "climbing & descending"),
    ("Calves", "every single step"),
    ("Hamstrings", "every step, knee control"),
    ("Shins", "descending"),
    ("Core & lower back", "balance & backpack"),
    ("Shoulders & arms", "straps & trekking poles"),
]

NUMERIC = ["slope", "stride", "period", "lean", "poles"]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scene_at(time):
    for i, s in enumerate(SCENES):
        if time < s["start"] + s["dur"] or i == len(SCENES) - 1:
            return i, time - s["start"]


def params_at(time):
    """Walking parameters and muscle intensities, eased between scenes."""
    i, u = scene_at(time)
    cur = SCENES[i]
    prev = SCENES[max(i - 1, 0)]
    k = smooth(u / 1.2)
    p = {n: prev[n] + (cur[n] - prev[n]) * k for n in NUMERIC}
    km = smooth(u / 0.8)
    p["muscles"] = {m: prev["muscles"].get(m, 0) * (1 - km) + cur["muscles"].get(m, 0) * km
                    for m in MUSCLES}
    return p


# ---------------------------------------------------------------- geometry --

HX, GY = 500, 610  # hiker hip x, ground height under the hiker
THIGH, SHIN = 88, 86
TORSO, NECK, HEAD_R = 118, 14, 25
UPPER_ARM, FOREARM = 62, 58
ANKLE_H = 12


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, k):
    return (a[0] * k, a[1] * k)


def norm(a):
    length = math.hypot(*a) or 1.0
    return (a[0] / length, a[1] / length)


def lerp(a, b, k):
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)


def from_angle(deg, length):
    """Vector pointing down, rotated forward (+x) by deg."""
    r = math.radians(deg)
    return (math.sin(r) * length, math.cos(r) * length)


def front_normal(a, b):
    """Unit normal of segment a->b that points toward the hiker's front (+x)."""
    d = norm(sub(b, a))
    n = (d[1], -d[0])
    return n if n[0] >= 0 else mul(n, -1)


def ground_y(x, slope):
    return GY + slope * (x - HX)


def ik_knee(hip, ankle):
    d = math.hypot(*sub(ankle, hip))
    d = min(d, (THIGH + SHIN) * 0.999)
    a = math.acos(max(-1.0, min(1.0, (THIGH ** 2 + d ** 2 - SHIN ** 2) / (2 * THIGH * d))))
    base = math.atan2(ankle[1] - hip[1], ankle[0] - hip[0])
    options = [(hip[0] + THIGH * math.cos(base + s * a), hip[1] + THIGH * math.sin(base + s * a))
               for s in (1, -1)]
    return max(options, key=lambda p: p[0])  # knees bend forward


def leg_pose(q, p, hip):
    q %= 1.0
    stride, slope = p["stride"], p["slope"]
    if q < 0.6:  # stance: foot planted, slides back relative to the hip
        s = q / 0.6
        dx = stride / 2 - stride * s
        lift = 6 * smooth((s - 0.75) / 0.25)  # heel rises before toe-off
        pitch = -14 * smooth((s - 0.7) / 0.3)
        act = math.sin(math.pi * s)
    else:  # swing
        s = (q - 0.6) / 0.4
        dx = -stride / 2 + stride * (0.5 - 0.5 * math.cos(math.pi * s))
        lift = 26 * math.sin(math.pi * s) + 6 * (1 - s)
        pitch = -14 * (1 - s) + 10 * math.sin(math.pi * s)
        act = 0.15
    x = HX + dx
    ankle = (x, ground_y(x, slope) - ANKLE_H - lift)
    knee = ik_knee(hip, ankle)
    return dict(hip=hip, knee=knee, ankle=ankle, pitch=pitch, act=act, fwd=dx / (stride / 2))


def pose(phase, p):
    slope = p["slope"]
    drop = 16 * min(abs(slope) / 0.2, 1.0)
    bob = 4 * math.cos(4 * math.pi * phase)
    hip = (HX, ground_y(HX, slope) - ANKLE_H - 160 + drop + bob)
    near = leg_pose(phase, p, hip)
    far = leg_pose(phase + 0.5, p, hip)
    lean = p["lean"]
    shoulder = add(hip, mul(from_angle(180 - lean, 1), TORSO))
    neck_top = add(shoulder, mul(from_angle(180 - lean * 0.6, 1), NECK))
    head = add(neck_top, mul(from_angle(180 - lean * 0.4, 1), HEAD_R - 3))

    def arm(fwd):
        swing = -fwd  # arms swing opposite to the leg on the same side
        upper = swing * 24 + lean * 0.3 + p["poles"] * 12
        elbow = add(shoulder, from_angle(upper, UPPER_ARM))
        bend = 18 + 14 * max(0.0, swing) + p["poles"] * 30
        hand = add(elbow, from_angle(upper + bend, FOREARM))
        return dict(shoulder=shoulder, elbow=elbow, hand=hand)

    return dict(hip=hip, shoulder=shoulder, neck_top=neck_top, head=head, lean=lean,
                near=near, far=far, near_arm=arm(near["fwd"]), far_arm=arm(far["fwd"]))


# ---------------------------------------------------------------- drawing --

FIG_K = 1.18  # hiker is drawn scaled by this about (HX, GY)
_fig_scale = [1.0]


def fig(pt):
    k = FIG_K
    return (HX + (pt[0] - HX) * k, GY + (pt[1] - GY) * k)


def S(pts):
    k = _fig_scale[0]
    return [((HX + (x - HX) * k) * SS, (GY + (y - GY) * k) * SS) for x, y in pts]


def circle(d, c, r, fill):
    d.ellipse(S([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]), fill=fill)


def capsule(a, b, wa, wb):
    n = norm(sub(b, a))
    n = (-n[1], n[0])
    return [add(a, mul(n, wa)), add(b, mul(n, wb)), sub(b, mul(n, wb)), sub(a, mul(n, wa))]


def draw_capsule(d, a, b, wa, wb, color):
    d.polygon(S(capsule(a, b, wa, wb)), fill=color)
    circle(d, a, wa, color)
    circle(d, b, wb, color)


def lens(a, b, side, t0, t1, w, off, n=16):
    """Bulging muscle shape along segment a->b on the given side."""
    outer, inner = [], []
    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        s = math.sin(math.pi * i / n)
        c = lerp(a, b, t)
        outer.append(add(c, mul(side, off + w * s)))
        inner.append(add(c, mul(side, off - w * 0.5 * s)))
    return outer + inner[::-1]


def ellipse_poly(c, rx, ry, angle=0.0, n=24):
    ca, sa = math.cos(angle), math.sin(angle)
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        x, y = rx * math.cos(th), ry * math.sin(th)
        pts.append((c[0] + x * ca - y * sa, c[1] + x * sa + y * ca))
    return pts


NEUTRAL = (74, 86, 120)
AMBER = (255, 150, 50)
HOT = (238, 52, 38)


def heat(i):
    if i <= 0.5:
        k = i / 0.5
        return tuple(int(NEUTRAL[j] + (AMBER[j] - NEUTRAL[j]) * k) for j in range(3))
    k = (i - 0.5) / 0.5
    return tuple(int(AMBER[j] + (HOT[j] - AMBER[j]) * k) for j in range(3))


def dim(c, k):
    return tuple(int(v * k) for v in c)


def muscle_shapes(P):
    """[(muscle, polygon, leg_activation, near?)] plus anchor points for labels."""
    shapes, anchors = [], {}
    torso_front = front_normal(P["hip"], P["shoulder"])
    torso_back = mul(torso_front, -1)
    for side_name in ("far", "near"):
        leg = P[side_name]
        hip, knee, ankle, act = leg["hip"], leg["knee"], leg["ankle"], leg["act"]
        fn = front_normal(hip, knee)
        bn = mul(fn, -1)
        sfn = front_normal(knee, ankle)
        sbn = mul(sfn, -1)
        near = side_name == "near"
        gc = add(add(hip, mul(torso_back, 12)), (0, 4))
        shapes.append(("glutes", ellipse_poly(gc, 19, 23, math.radians(-P["lean"])), act, near))
        shapes.append(("quads", lens(hip, knee, fn, 0.1, 0.95, 11, 1), act, near))
        shapes.append(("hamstrings", lens(hip, knee, bn, 0.15, 0.92, 10, 1), act, near))
        shapes.append(("hip_flexors", lens(add(hip, mul(torso_front, 4)), knee, fn, 0.0, 0.3, 8, 9), act, near))
        shapes.append(("calves", lens(knee, ankle, sbn, 0.05, 0.7, 10, 1), act, near))
        shapes.append(("shins", lens(knee, ankle, sfn, 0.1, 0.8, 5, 2), act, near))
        if near:
            anchors["glutes"] = add(gc, mul(torso_back, 8))
            anchors["quads"] = add(lerp(hip, knee, 0.55), mul(fn, 12))
            anchors["hamstrings"] = add(lerp(hip, knee, 0.55), mul(bn, 11))
            anchors["hip_flexors"] = add(lerp(hip, knee, 0.12), mul(fn, 15))
            anchors["calves"] = add(lerp(knee, ankle, 0.32), mul(sbn, 14))
            anchors["shins"] = add(lerp(knee, ankle, 0.45), mul(sfn, 8))
    hip, sh = P["hip"], P["shoulder"]
    breath = 0.5 + 0.5 * math.sin(2 * math.pi * P["phase"] * 2)
    shapes.append(("core", lens(hip, sh, torso_front, 0.08, 0.72, 10, 13), breath, True))
    shapes.append(("lower_back", lens(hip, sh, torso_back, 0.04, 0.45, 9, 12), breath, True))
    shapes.append(("shoulders", lens(hip, sh, torso_back, 0.72, 1.02, 9, 13), breath, True))
    shapes.append(("shoulders", ellipse_poly(add(sh, mul(norm(sub(P["near_arm"]["elbow"], sh)), 10)), 13, 11), breath, True))
    anchors["core"] = add(lerp(hip, sh, 0.4), mul(torso_front, 22))
    anchors["lower_back"] = add(lerp(hip, sh, 0.22), mul(torso_back, 19))
    anchors["shoulders"] = add(lerp(hip, sh, 0.9), mul(torso_back, 18))
    for side_name in ("far_arm", "near_arm"):
        arm = P[side_name]
        fn = front_normal(arm["shoulder"], arm["elbow"])
        near = side_name == "near_arm"
        shapes.append(("arms", lens(arm["shoulder"], arm["elbow"], mul(fn, -1), 0.25, 0.95, 7, 2), breath, near))
        shapes.append(("arms", lens(arm["shoulder"], arm["elbow"], fn, 0.3, 0.9, 6, 2), breath, near))
        if near:
            anchors["arms"] = add(lerp(arm["shoulder"], arm["elbow"], 0.6), mul(fn, -9))
    return shapes, anchors


def intensity(muscles, name, act):
    base = muscles.get(name, 0.0)
    return base * (0.55 + 0.45 * act)


BODY = (46, 56, 84)
BODY_FAR = (32, 39, 60)
SKIN = (233, 190, 160)
PACK = (36, 120, 104)
PACK_DARK = (26, 88, 76)
SHOE = (70, 52, 40)


def draw_shoe(d, leg, slope, color):
    a = leg["ankle"]
    u = norm((1, slope))
    ang = math.radians(leg["pitch"])
    u = (u[0] * math.cos(ang) + u[1] * math.sin(ang), -u[0] * math.sin(ang) + u[1] * math.cos(ang))
    dn = (-u[1], u[0])
    pts = [add(a, add(mul(u, -12), mul(dn, -6))), add(a, add(mul(u, 12), mul(dn, -3))),
           add(a, add(mul(u, 30), mul(dn, 5))), add(a, add(mul(u, 30), mul(dn, 12))),
           add(a, add(mul(u, -13), mul(dn, 12)))]
    d.polygon(S(pts), fill=color)


def draw_hiker(d, P, p, shapes, muscles):
    def leg(L, body, mk):
        draw_capsule(d, L["hip"], L["knee"], 15, 11, body)
        draw_capsule(d, L["knee"], L["ankle"], 11, 7, body)
        draw_shoe(d, L, p["slope"], dim(SHOE, mk))

    def arm(A, body):
        draw_capsule(d, A["shoulder"], A["elbow"], 9, 7, body)
        draw_capsule(d, A["elbow"], A["hand"], 7, 6, body)
        circle(d, A["hand"], 7, dim(SKIN, 0.85))

    def pole(A, alpha):
        if alpha <= 0.01:
            return
        h = A["hand"]
        tip_x = h[0] + 26
        tip = (tip_x, ground_y(tip_x, p["slope"]))
        top = add(h, mul(norm(sub(h, tip)), 12))
        d.line(S([top, tip]), fill=(200, 205, 215, int(255 * alpha)), width=5 * SS)

    def muscles_of(near_flag, names, k=1.0):
        for name, poly, act, near in shapes:
            if near == near_flag and name in names:
                i = intensity(muscles, name, act)
                d.polygon(S(poly), fill=dim(heat(i), k))

    legs = ("glutes", "quads", "hamstrings", "hip_flexors", "calves", "shins")
    arms_m = ("arms",)

    # far side (darker, behind the body)
    pole(P["far_arm"], p["poles"] * 0.7)
    arm(P["far_arm"], BODY_FAR)
    muscles_of(False, arms_m, 0.7)
    leg(P["far"], BODY_FAR, 0.7)
    muscles_of(False, legs, 0.7)

    # backpack behind the torso
    hip, sh = P["hip"], P["shoulder"]
    back = mul(front_normal(hip, sh), -1)
    up = norm(sub(sh, hip))
    pc = add(lerp(hip, sh, 0.62), mul(back, 34))
    pack = [add(pc, add(mul(up, sy * 56), mul(back, sx * 24))) for sx, sy in ((-1, 1), (1, 0.9), (1, -0.8), (-1, -1))]
    d.polygon(S(pack), fill=PACK)
    d.polygon(S([add(pc, add(mul(up, sy * 30), mul(back, sx * 26))) for sx, sy in ((0, 0.6), (1, 0.5), (1, -0.6), (0, -0.7))]), fill=PACK_DARK)

    # torso, neck, head
    draw_capsule(d, hip, sh, 21, 22, BODY)
    draw_capsule(d, sh, P["neck_top"], 8, 8, dim(SKIN, 0.9))
    for name in ("core", "lower_back", "shoulders"):
        muscles_of(True, (name,))
    # backpack strap over the shoulder
    d.line(S([add(lerp(hip, sh, 0.98), mul(back, 6)), add(lerp(hip, sh, 0.82), mul(back, -14)),
              add(lerp(hip, sh, 0.45), mul(back, -10)), add(lerp(hip, sh, 0.4), mul(back, 14))]),
           fill=PACK_DARK, width=6 * SS, joint="curve")

    hd = P["head"]
    circle(d, hd, HEAD_R, SKIN)
    # hat: brim + crown
    brim_y = hd[1] - 8
    d.polygon(S([(hd[0] - 30, brim_y), (hd[0] + 36, brim_y - 3), (hd[0] + 34, brim_y + 3), (hd[0] - 30, brim_y + 5)]), fill=(150, 110, 60))
    d.chord(S([(hd[0] - 24, hd[1] - 34), (hd[0] + 24, hd[1] + 10)]), 180, 360, fill=(170, 125, 70))
    circle(d, (hd[0] + 14.5, hd[1] + 0.5), 2.5, (40, 40, 50))

    # near side
    leg(P["near"], BODY, 1.0)
    muscles_of(True, legs)
    arm(P["near_arm"], BODY)
    muscles_of(True, arms_m)
    pole(P["near_arm"], p["poles"])


# ------------------------------------------------------------- background --

PW = 2 * W  # period of the scrolling background layers


def periodic(x, terms):
    return sum(a * np.abs(np.sin(np.pi * x * k / PW + ph)) ** e for a, k, ph, e in terms)


def build_background():
    w2, h2 = W * SS, H * SS
    yy = np.linspace(0, 1, h2)[:, None]
    top, bottom = np.array([110, 170, 220]), np.array([240, 228, 205])
    sky = (top + (bottom - top) * np.clip(yy * 1.25, 0, 1)[..., None]).astype(np.uint8)
    sky = np.repeat(sky, w2, axis=1)
    sky_img = Image.fromarray(sky.reshape(h2, w2, 3))
    sun = Image.new("L", (w2, h2), 0)
    ImageDraw.Draw(sun).ellipse([1040 * SS, 80 * SS, 1130 * SS, 170 * SS], fill=255)
    halo = sun.filter(ImageFilter.GaussianBlur(40 * SS))
    sky_img.paste((255, 236, 180), (0, 0), halo.point(lambda v: v // 2))
    sky_img.paste((255, 244, 210), (0, 0), sun)

    layers = []
    xs = np.arange((PW + W) * SS) / SS
    ys = np.arange(H * SS)[:, None] / SS
    specs = [
        # color, ridge height function, parallax factor, snow line
        ((150, 168, 196), 470 - periodic(xs, [(150, 3, 0.3, 1.6), (70, 7, 1.1, 2.0), (30, 17, 0.4, 1.0)]), 0.06, 345),
        ((100, 132, 132), 540 - periodic(xs, [(80, 2, 1.7, 1.2), (40, 5, 0.2, 1.5), (14, 23, 0.9, 1.0)]), 0.18, None),
    ]
    for color, ridge, factor, snow in specs:
        mask = (ys >= ridge[None, :]).astype(np.uint8) * 255
        rgb = np.empty(mask.shape + (3,), np.uint8)
        rgb[:] = color
        if snow is not None:
            cap = (ys < snow) & (ys < ridge[None, :] + 30)
            rgb[cap] = (240, 244, 250)
        img = Image.fromarray(rgb)
        img.putalpha(Image.fromarray(mask))
        layers.append((img, factor))
    # pine tree line
    rng = np.random.default_rng(7)
    trees = Image.new("RGBA", ((PW + W) * SS, H * SS), (0, 0, 0, 0))
    td = ImageDraw.Draw(trees)
    base = 570
    td.rectangle([0, (base - 4) * SS, (PW + W) * SS, H * SS], fill=(60, 96, 80))
    x = 0.0
    while x < PW:
        h = rng.uniform(40, 85)
        wdt = h * 0.42
        for off in (0, PW):
            cx = x + off
            if cx - wdt > PW + W:
                continue
            td.polygon(S([(cx, base - h), (cx + wdt, base), (cx - wdt, base)]), fill=(52, 86, 72))
        x += rng.uniform(18, 40)
    layers.append((trees, 0.4))
    return sky_img, layers


rng_g = np.random.default_rng(3)
GROUND_ITEMS = [(float(x), rng_g.choice(["tuft", "rock", "tuft", "flower"]), float(rng_g.uniform(0.7, 1.3)))
                for x in np.cumsum(rng_g.uniform(40, 110, 400))]


def draw_ground(d, slope, scroll):
    gl, gr = ground_y(0, slope), ground_y(W, slope)
    d.polygon(S([(0, gl), (W, gr), (W, H), (0, H)]), fill=(118, 146, 84))
    d.polygon(S([(0, gl + 26), (W, gr + 26), (W, H), (0, H)]), fill=(98, 124, 70))
    d.polygon(S([(0, gl - 2), (W, gr - 2), (W, gr + 16), (0, gl + 16)]), fill=(186, 158, 116))
    d.line(S([(0, gl - 2), (W, gr - 2)]), fill=(140, 116, 82), width=2 * SS)
    span = GROUND_ITEMS[-1][0]
    for wx, kind, k in GROUND_ITEMS:
        x = (wx - scroll) % span - 60
        if x < -40 or x > W + 40:
            continue
        y = ground_y(x, slope)
        if kind == "rock":
            d.ellipse(S([(x - 9 * k, y + 4 - 7 * k), (x + 9 * k, y + 8)]), fill=(130, 124, 116))
        elif kind == "flower":
            d.line(S([(x, y + 22), (x, y + 10)]), fill=(70, 110, 60), width=2 * SS)
            d.ellipse(S([(x - 4, y + 6), (x + 4, y + 14)]), fill=(240, 220, 90))
        else:
            for dx in (-6, 0, 6):
                d.line(S([(x + dx * k, y + 26), (x + dx * 1.6 * k, y + 14 - 4 * k)]), fill=(80, 116, 62), width=2 * SS)


# --------------------------------------------------------------------- UI --

ACCENT = (255, 120, 60)


def fade_window(u, dur, t_in, length=0.5, out=0.45):
    return smooth((u - t_in) / length) * smooth((dur - u) / out)


def rounded(d, box, r, fill):
    d.rounded_rectangle([v * SS for v in box], radius=r * SS, fill=fill)


def text(d, xy, s, size, fill, bold=False, anchor="la"):
    d.text((xy[0] * SS, xy[1] * SS), s, font=font(size, bold), fill=fill, anchor=anchor)


def text_w(s, size, bold=False):
    return font(size, bold).getlength(s) / SS


def draw_label(d, anchor, title, subtitle, side, y, a, color):
    bw, bh = 300, 66
    bx = 36 if side == "L" else W - 36 - bw
    if a <= 0.01:
        return
    edge = (bx + bw, y + bh / 2) if side == "L" else (bx, y + bh / 2)
    la = int(230 * a)
    d.line(S([edge, anchor]), fill=(255, 255, 255, la), width=2 * SS)
    r = 6
    d.ellipse([(anchor[0] - r) * SS, (anchor[1] - r) * SS, (anchor[0] + r) * SS, (anchor[1] + r) * SS],
              fill=(255, 255, 255, la), outline=color + (la,), width=2 * SS)
    rounded(d, (bx, y, bx + bw, y + bh), 12, (20, 26, 40, int(215 * a)))
    rounded(d, (bx, y, bx + 7, y + bh), 4, color + (int(255 * a),))
    text(d, (bx + 20, y + 10), title, 23, (255, 255, 255, int(255 * a)), bold=True)
    text(d, (bx + 20, y + 40), subtitle, 16, (210, 218, 232, int(255 * a)))


def draw_header(d, idx, u, time):
    sc = SCENES[idx]
    # overall progress bar with scene ticks
    d.rectangle(S([(0, 0), (W, 6)]), fill=(0, 0, 0, 90))
    d.rectangle(S([(0, 0), (W * time / TOTAL, 6)]), fill=ACCENT + (255,))
    for s in SCENES[1:]:
        x = W * s["start"] / TOTAL
        d.rectangle(S([(x - 1, 0), (x + 1, 6)]), fill=(255, 255, 255, 160))
    if not sc["heading"]:
        return
    a = fade_window(u, sc["dur"], 0.1, 0.5, 0.35)
    if a <= 0.01:
        return
    step = f"{idx} / {len(SCENES) - 1}"
    tw = text_w(sc["heading"], 26, True) + text_w(step, 16) + 64
    rounded(d, (36, 26, 36 + tw, 74), 24, (20, 26, 40, int(210 * a)))
    text(d, (58, 50), step, 16, ACCENT + (int(255 * a),), anchor="lm")
    text(d, (58 + text_w(step, 16) + 18, 50), sc["heading"], 26, (255, 255, 255, int(255 * a)), True, "lm")


def draw_caption(d, sc, u):
    if not sc["caption"]:
        return
    a = fade_window(u, sc["dur"], 0.6, 0.6, 0.4)
    if a <= 0.01:
        return
    size = 25
    tw = text_w(sc["caption"], size, True)
    rounded(d, (W / 2 - tw / 2 - 26, 642, W / 2 + tw / 2 + 26, 694), 26, (20, 26, 40, int(215 * a)))
    text(d, (W / 2, 668), sc["caption"], size, (255, 255, 255, int(255 * a)), True, "mm")


def draw_title(d, u, dur):
    a = smooth(u / 0.4) * smooth((dur - u) / 0.7) if u > 0 else 1.0
    a = max(a, 1 - smooth(u / 0.4))  # fully covered on the very first frame
    d.rectangle(S([(0, 0), (W, H)]), fill=(14, 20, 34, int(190 * a)))
    ta = smooth((u - 0.2) / 0.6) * smooth((dur - u) / 0.7)
    text(d, (W / 2, 300), "Which muscles work", 64, (255, 255, 255, int(255 * ta)), True, "mm")
    text(d, (W / 2, 378), "on a hike?", 64, ACCENT + (int(255 * ta),), True, "mm")
    sa = smooth((u - 0.9) / 0.6) * smooth((dur - u) / 0.7)
    text(d, (W / 2, 450), "Follow the glow: the hotter the color, the harder the muscle works",
         22, (220, 228, 240, int(255 * sa)), anchor="mm")


def draw_recap(d, u, dur):
    a = smooth((u - 0.3) / 0.6)
    if a <= 0.01:
        return
    x0, y0, x1 = 780, 104, W - 36
    rows_h = 58
    y1 = y0 + 70 + rows_h * len(RECAP_ROWS)
    rounded(d, (x0, y0, x1, y1), 18, (20, 26, 40, int(220 * a)))
    text(d, (x0 + 26, y0 + 22), "Your hiking muscles", 27, (255, 255, 255, int(255 * a)), True)
    for i, (name, where) in enumerate(RECAP_ROWS):
        ra = a * smooth((u - 0.6 - i * 0.25) / 0.4)
        y = y0 + 76 + i * rows_h
        c = heat(0.85 - 0.06 * i)
        d.ellipse(S([(x0 + 26, y + 6), (x0 + 44, y + 24)]), fill=c + (int(255 * ra),))
        text(d, (x0 + 60, y + 1), name, 21, (255, 255, 255, int(255 * ra)), True)
        text(d, (x0 + 60, y + 28), where, 16, (205, 214, 230, int(255 * ra)))
    # gentle fade to dark at the very end
    end = smooth((u - (dur - 0.8)) / 0.8)
    if end > 0:
        d.rectangle(S([(0, 0), (W, H)]), fill=(10, 14, 24, int(255 * end)))


# ------------------------------------------------------------------ render --

class Renderer:
    def __init__(self):
        self.sky, self.layers = build_background()
        self.phase = 0.0
        self.scroll = 0.0
        self.time = 0.0

    def advance(self, dt):
        p = params_at(self.time)
        self.phase = (self.phase + dt / p["period"]) % 1.0
        self.scroll += dt * p["stride"] / (0.6 * p["period"])
        self.time += dt

    def frame(self):
        time = self.time
        p = params_at(time)
        idx, u = scene_at(time)
        sc = SCENES[idx]

        img = self.sky.copy()
        for layer, factor in self.layers:
            off = int((self.scroll * factor) % PW * SS)
            crop = layer.crop((off, 0, off + W * SS, H * SS))
            img.paste(crop, (0, 0), crop)
        d = ImageDraw.Draw(img, "RGBA")
        draw_ground(d, p["slope"], self.scroll)

        P = pose(self.phase, p)
        P["phase"] = self.phase
        shapes, anchors = muscle_shapes(P)
        _fig_scale[0] = FIG_K
        draw_hiker(d, P, p, shapes, p["muscles"])
        _fig_scale[0] = 1.0
        anchors = {m: fig(pt) for m, pt in anchors.items()}

        # bloom around working muscles
        gw, gh = W // GLOW_DIV, H // GLOW_DIV
        glow = Image.new("RGB", (gw, gh), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        for name, poly, act, near in shapes:
            i = intensity(p["muscles"], name, act)
            if i < 0.05:
                continue
            k = i * (1.0 if near else 0.55)
            gd.polygon([(x / GLOW_DIV, y / GLOW_DIV) for x, y in map(fig, poly)], fill=tuple(int(c * k) for c in heat(i)))
        glow = glow.filter(ImageFilter.GaussianBlur(4)).resize(img.size, Image.BILINEAR)
        img = Image.fromarray(np.minimum(np.asarray(img, np.uint16) + np.asarray(glow, np.uint16), 255).astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")

        for k, (m, title, subtitle, side, y) in enumerate(sc["labels"]):
            a = fade_window(u, sc["dur"], 1.0 + 0.45 * k, 0.5, 0.4)
            draw_label(d, anchors[m], title, subtitle, side, y, a, heat(sc["muscles"].get(m, 0.6)))
        draw_header(d, idx, u, time)
        draw_caption(d, sc, u)
        if sc["key"] == "title":
            draw_title(d, u, sc["dur"])
        if sc["key"] == "recap":
            draw_recap(d, u, sc["dur"])
        return img.reduce(SS)


def render_video():
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(OUT_VIDEO)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    r = Renderer()
    n = int(round(TOTAL * FPS))
    poster_frame = int((SCENES[-1]["start"] + 4.0) * FPS)
    for f in range(n):
        img = r.frame()
        if f == poster_frame:
            img.save(OUT_POSTER)
        proc.stdin.write(img.tobytes())
        r.advance(1 / FPS)
        if f % FPS == 0:
            print(f"\r{f / FPS:5.1f}s / {TOTAL:.1f}s", end="", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"\nwrote {OUT_VIDEO.relative_to(ROOT)} and {OUT_POSTER.relative_to(ROOT)}")


def render_previews(times, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Renderer()
    for target in sorted(times):
        while r.time + 1e-9 < target:
            r.advance(1 / FPS)
        path = out_dir / f"frame_{target:05.1f}.png"
        r.frame().save(path)
        print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=float, nargs="+", help="render preview PNGs at these times (s)")
    ap.add_argument("--out", type=Path, default=Path("preview"), help="preview output directory")
    args = ap.parse_args()
    if args.frames:
        render_previews(args.frames, args.out)
    else:
        render_video()
