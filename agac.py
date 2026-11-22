#!/usr/bin/env python3
# ASCII Noel agaci - terminal surumu.  Gereken: python3 + numpy
# Calistir:  python agac.py      Cikis: Ctrl+C
import sys, os, time, math, shutil
import numpy as np

TAU = math.tau
Y0, Y1 = 0.42, 2.55            # yaprakli govdenin alt / ust yuksekligi
CYCLE = 22.0                   # saniye: buyume + tam tur, sonra basa doner
FPS = 30
RAMP = " .,:;+=xX#%@"          # harf/karakter parlaklik sirasi

NEEDLE, TRUNK, BALL, GARLAND, LIGHT, STAR, GIFT, GROUND = range(8)
AMB = np.array([0.42, 0.35, 0.25, 0.62, 0, 0, 0.5, 0.35])
SPEC = np.array([0.0, 0.0, 0.9, 0.5, 0, 0, 0.25, 0.0])


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def envelope(y):
    f = np.clip((y - Y0) / (Y1 - Y0), 0, 1)
    return 0.03 + 0.84 * (1 - f) ** 1.08


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class Parts:
    def __init__(s):
        s.P, s.N, s.C, s.K, s.T, s.PH = [], [], [], [], [], []

    def add(s, p, n, col, kind, t0, ph=0.0):
        m = len(p)
        s.P.append(np.asarray(p, float))
        s.N.append(unit(np.asarray(n, float)))
        s.C.append(np.broadcast_to(np.asarray(col, float), (m, 3)))
        s.K.append(np.full(m, kind))
        s.T.append(np.broadcast_to(np.asarray(t0, float), (m,)))
        s.PH.append(np.broadcast_to(np.asarray(ph, float), (m,)))

    def done(s):
        return [np.concatenate(v) for v in (s.P, s.N, s.C, s.K, s.T, s.PH)]


def sphere(n):
    i = np.arange(n) + 0.5
    y = 1 - 2 * i / n
    r = np.sqrt(1 - y * y)
    th = math.pi * (3 - math.sqrt(5)) * i
    return np.stack([r * np.cos(th), y, r * np.sin(th)], 1)


def build():
    rng = np.random.default_rng(7)
    S = Parts()

    # --- igne yapraklar: sarkan dallarin halkalari
    base = np.array([0.07, 0.30, 0.13])
    tiers = np.linspace(Y0 + 0.02, Y1 - 0.04, 24)
    for i, y in enumerate(tiers):
        R = envelope(y)
        f = (y - Y0) / (Y1 - Y0)
        nb = int(7 + 11 * (1 - f))
        for b in range(nb):
            a = TAU * b / nb + i * 0.618 * TAU + rng.normal(0, 0.12)
            L = R * rng.uniform(0.82, 1.06)
            n = int(120 + 700 * L)
            s = rng.random(n) ** 0.65
            d = np.array([math.cos(a), 0, math.sin(a)])
            side = np.array([-math.sin(a), 0, math.cos(a)])
            spread = (0.035 + 0.11 * s) * rng.normal(0, 1, n)
            yy = y + 0.07 * s * L - 0.22 * (s * L) ** 2 + rng.normal(0, 0.03, n)
            p = d[None] * (s * L)[:, None] + side[None] * spread[:, None]
            p[:, 1] = yy
            nrm = d[None] * 0.8 + np.array([0, 0.6, 0]) + rng.normal(0, 0.25, (n, 3))
            ao = 0.35 + 0.65 * s ** 1.3
            tone = rng.uniform(0.75, 1.1, n)
            col = (base[None] + rng.random((n, 3)) * [0.06, 0.18, 0.05]) * (ao * tone)[:, None]
            snow = (s > 0.45) & (rng.random(n) < 0.05)
            col[snow] = [0.92, 0.95, 1.0]
            t0 = 0.3 + 5.6 * ease(f) + rng.random(n) * 0.3
            S.add(p, nrm, col, NEEDLE, t0)

    # --- govde
    h, a = np.meshgrid(np.linspace(-0.05, 0.6, 50), np.linspace(0, TAU, 28, endpoint=False))
    h, a = h.ravel(), a.ravel() + rng.normal(0, 0.05, h.size)
    p = np.stack([0.085 * np.cos(a), h, 0.085 * np.sin(a)], 1)
    nn = np.stack([np.cos(a), 0 * a, np.sin(a)], 1)
    bark = 0.75 + 0.25 * np.sin(h * 90 + a * 3)
    S.add(p, nn, np.array([0.42, 0.26, 0.14])[None] * bark[:, None], TRUNK, 4.0)

    # --- tinsel: spiral seklinde sarkan altin sarmal
    u = np.linspace(0, 1, 2600)
    turns = 3.6
    yg = Y0 + 0.15 + (Y1 - Y0 - 0.45) * u
    ag = TAU * turns * u + 0.9
    swag = -0.06 * np.abs(np.sin(ag * 2.5))
    r = envelope(yg) * 0.97 + rng.normal(0, 0.012, u.size)
    p = np.stack([r * np.cos(ag), yg + swag, r * np.sin(ag)], 1)
    S.add(p, np.stack([np.cos(ag), 0.3 + 0 * ag, np.sin(ag)], 1),
          [1.0, 0.78, 0.25], GARLAND, 7.0 + 5.0 * u)

    # --- toplar
    cols = np.array([[0.85, 0.05, 0.07], [0.95, 0.70, 0.15], [0.10, 0.25, 0.90], [0.80, 0.82, 0.90]])
    for k in range(40):
        y = rng.uniform(Y0 + 0.2, Y1 - 0.35)
        ang = rng.uniform(0, TAU)
        rr = envelope(y) * rng.uniform(0.55, 0.95)
        c = np.array([rr * math.cos(ang), y - 0.05, rr * math.sin(ang)])
        sp = sphere(90)
        S.add(c + sp * 0.065, sp, cols[rng.choice(4, p=[.5, .25, .15, .1])], BALL, 6.0 + rng.random() * 4)

    # --- isiklar (yanip soner)
    lc = np.array([[1.0, 0.85, 0.5], [1.0, 0.25, 0.2], [0.4, 0.7, 1.0], [1.0, 0.9, 0.7]])
    for k in range(170):
        y = rng.uniform(Y0 + 0.15, Y1 - 0.1)
        ang = rng.uniform(0, TAU)
        rr = envelope(y) * rng.uniform(0.5, 0.98)
        c = np.array([rr * math.cos(ang), y, rr * math.sin(ang)])
        S.add(c + rng.normal(0, 0.012, (3, 3)), np.ones((3, 3)), lc[rng.integers(4)], LIGHT,
              8.0 + 8.0 * k / 170, rng.uniform(0, TAU))

    # --- yildiz
    top = np.array([0, Y1 + 0.08, 0])
    S.add(top + sphere(60) * 0.07, np.ones((60, 3)), [1.0, 0.85, 0.3], STAR, 7.0)
    for k in range(8):
        ang = k * TAU / 8
        ln = 0.28 if k % 2 == 0 else 0.15
        q = np.linspace(0.05, ln, 14)
        S.add(top + np.stack([q * np.cos(ang), q * np.sin(ang), 0 * q], 1),
              np.ones((14, 3)), [1.0, 0.8, 0.3], STAR, 7.0)

    # --- hediye kutulari
    def box(cx, cz, size, col, rib, yaw, t0):
        sx, sy, sz = size
        pts, nrm, cl = [], [], []
        for ax in range(3):
            for sg in (-1, 1):
                m = 450
                uv = rng.uniform(-1, 1, (m, 2))
                q = np.zeros((m, 3))
                o = [j for j in range(3) if j != ax]
                q[:, o[0]] = uv[:, 0]
                q[:, o[1]] = uv[:, 1]
                q[:, ax] = sg
                nv = np.zeros((m, 3))
                nv[:, ax] = sg
                isr = (np.abs(uv[:, 0]) < 0.14) | (np.abs(uv[:, 1]) < 0.14)
                c = np.where(isr[:, None], np.array(rib)[None], np.array(col)[None])
                pts.append(q * [sx, sy, sz])
                nrm.append(nv)
                cl.append(c)
        pts, nrm, cl = np.concatenate(pts), np.concatenate(nrm), np.concatenate(cl)
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        Rm = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
        pts, nrm = pts @ Rm.T, nrm @ Rm.T
        pts = pts * 1.35 + [cx, sy * 1.35, cz]
        S.add(pts, nrm, cl, GIFT, t0 + rng.random(len(pts)) * 0.4)

    box(0.95, 0.40, (0.22, 0.17, 0.18), [0.8, 0.07, 0.08], [0.95, 0.8, 0.3], 0.4, 10.0)
    box(-0.9, 0.55, (0.17, 0.22, 0.17), [0.1, 0.2, 0.8], [0.9, 0.9, 0.95], 1.1, 10.4)
    box(0.35, -1.05, (0.26, 0.15, 0.20), [0.95, 0.75, 0.2], [0.8, 0.07, 0.08], 2.2, 10.8)
    box(-0.55, -0.95, (0.18, 0.18, 0.18), [0.1, 0.55, 0.25], [0.95, 0.95, 0.95], 0.2, 11.2)
    box(1.25, -0.35, (0.14, 0.12, 0.14), [0.7, 0.2, 0.6], [0.95, 0.8, 0.3], 0.9, 11.6)

    # --- zemin: karli cizgili daire
    n = 5000
    rr = 2.0 * np.sqrt(rng.random(n))
    ang = rng.uniform(0, TAU, n)
    p = np.stack([rr * np.cos(ang), rng.normal(0, 0.01, n), rr * np.sin(ang)], 1)
    fade = np.clip(1.3 - rr / 2.0, 0.15, 1)
    S.add(p, np.tile([0, 1.0, 0], (n, 1)), np.array([0.16, 0.19, 0.28])[None] * fade[:, None], GROUND, 0.0)

    return S.done()


class Scene:
    def __init__(s):
        s.P, s.N, s.C, s.K, s.T, s.PH = build()
        s.P = s.P.astype(np.float32)
        s.emb = ((s.K == LIGHT) | (s.K == STAR)) * 0.15
        r = np.random.default_rng(3)
        s.sx = r.uniform(-3.2, 3.2, 260)
        s.sz = r.uniform(-2.6, 2.4, 260)
        s.sy0 = r.uniform(0, 4.2, 260)
        s.sv = r.uniform(0.3, 0.8, 260)
        s.sb = r.uniform(0.07, 0.30, 260)
        s.sph = r.uniform(0, TAU, 260)

    def render(s, t, rot, W, H):
        """Hucre ortusu uret: (renk HxWx3, dolu HxW, isik sayisi)."""
        vi = np.flatnonzero(s.T <= t)
        ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        cp, sp = math.cos(0.20), math.sin(0.20)       # kamera hafif yukaridan
        P = s.P[vi]
        x, y, z = P[:, 0], P[:, 1] - 1.4, P[:, 2]
        xr = ca * x + sa * z
        zr = -sa * x + ca * z
        y2 = y * cp - zr * sp
        z2 = y * sp + zr * cp
        D = 6.0
        zc = D - z2 - s.emb[vi]
        per = D / zc
        ry = 0.86 * H / 2.8
        cx_, cy_ = W / 2.0, H * 0.5 + 0.02 * H
        px = cx_ + xr * ry * 2.0 * per
        py = cy_ - y2 * ry * per
        ci = px.astype(np.int32)
        ri = py.astype(np.int32)
        ok = np.flatnonzero((ci >= 0) & (ci < W) & (ri >= 0) & (ri < H))
        cell = ri[ok] * W + ci[ok]
        order = np.argsort(-zc[ok])                    # uzaktan yakina, sonuncu kazanir
        win = np.full(H * W, -1, np.int64)
        win[cell[order]] = ok[order]
        cells = np.flatnonzero(win >= 0)
        gi = vi[win[cells]]                            # kazanan noktalar

        # sadece gorunen noktalar icin isik hesabi
        K, PH = s.K[gi], s.PH[gi]
        n0 = s.N[gi]
        nx = ca * n0[:, 0] + sa * n0[:, 2]
        nz0 = -sa * n0[:, 0] + ca * n0[:, 2]
        ny = n0[:, 1] * cp - nz0 * sp
        nz = n0[:, 1] * sp + nz0 * cp
        Nv = np.stack([nx, ny, nz], 1)
        Ld = unit(np.array([-0.4, 0.6, 0.7]))
        Hh = unit(Ld + np.array([0, 0, 1.0]))
        diff = np.clip(Nv @ Ld, 0, 1)
        spec = np.clip(Nv @ Hh, 0, 1) ** 30 * SPEC[K]
        C = s.C[gi]
        col = C * (AMB[K] + 0.9 * diff)[:, None] + spec[:, None]
        em = (K == LIGHT) | (K == STAR)
        tw = np.where(K == LIGHT, 0.55 + 0.45 * np.sin(3.0 * t + PH), 0.8 + 0.2 * np.sin(5 * t))
        col = np.where(em[:, None], C * tw[:, None] * 1.3, col)
        nlights = int(np.count_nonzero(s.K[vi] == LIGHT)) // 3

        buf = np.zeros((H * W, 3))
        fil = np.zeros(H * W, bool)
        buf[cells] = col
        fil[cells] = True

        # kar (gorunum uzayinda), sadece bos hucrelere
        sy = (s.sy0 - s.sv * t) % 4.2 - 1.8
        sxw = s.sx + 0.15 * np.sin(t * 0.9 + s.sph)
        zs = 6.0 - s.sz
        pxs = (cx_ + sxw * ry * 2.0 * (D / zs)).astype(int)
        pys = (cy_ - sy * ry * (D / zs)).astype(int)
        g = (pxs >= 0) & (pxs < W) & (pys >= 0) & (pys < H)
        sc_ = pys[g] * W + pxs[g]
        free = ~fil[sc_]
        sc_ = sc_[free]
        buf[sc_] = s.sb[g][free][:, None] * [0.9, 0.95, 1.0]
        fil[sc_] = True
        return buf.reshape(H, W, 3), fil.reshape(H, W), nlights


ESC = {}


def fg(r, g, b):
    k = (r, g, b)
    e = ESC.get(k)
    if e is None:
        e = ESC[k] = "\033[38;2;%d;%d;%dm" % k
    return e


def frame_text(sc, t, W, H, fps):
    ph = min(12, max(3, H // 4))
    Hr = H - ph - 2
    rot = (t * 18.0) % 360
    buf, fil, nl = sc.render(t, rot, W, Hr)

    lum = buf @ [0.3, 0.59, 0.11]
    ci = np.clip((np.clip(lum, 0, 1) ** 0.6 * (len(RAMP) - 1) * 1.15).astype(int) + 1, 1, len(RAMP) - 1)
    rgb = (np.clip(buf * 1.6, 0, 1) * 255).astype(int) >> 3 << 3
    rgb = np.clip(rgb, 0, 255)
    ci, rgb, fil = ci.tolist(), rgb.tolist(), fil.tolist()

    lines = []
    for r in range(Hr):
        out, prev = [], None
        cr, cc, cf = ci[r], rgb[r], fil[r]
        for c in range(W):
            if cf[c]:
                col = cc[c]
                k = (col[0], col[1], col[2])
                if k != prev:
                    out.append(fg(*k))
                    prev = k
                out.append(RAMP[cr[c]])
            else:
                out.append(" ")
        lines.append("".join(out))

    dim = "\033[38;2;90;100;125m"
    n_letters = sum(map(sum, fil))
    l1 = "999.2006.01.19 · 2026    rotation %06.2f°    lights %d" % (rot, nl)
    l2 = "letters %s    agac.py    running · %d f" % (format(n_letters, ","), fps)
    lines.append(dim + l1.ljust(W)[:W])
    lines.append(dim + l2.ljust(W)[:W])

    src = sc.src
    start = int(t * 6) % max(1, len(src) - ph)
    for j in range(ph):
        n = start + j
        txt = src[n] if n < len(src) else ""
        body = ("%3d  %s" % (n + 1, txt)).ljust(W)[:W]
        color = "\033[38;2;70;130;90m" if txt.lstrip().startswith("#") else "\033[38;2;95;110;150m"
        lines.append(color + body)
    return "\033[H" + "\n".join(lines) + "\033[0m"


def main():
    if os.name == "nt":
        os.system("")                       # Windows'ta ANSI renkleri ac
    sc = Scene()
    try:
        with open(__file__, encoding="utf-8") as f:
            sc.src = [l.rstrip("\n").replace("\t", "    ") for l in f]
    except Exception:
        sc.src = ["# agac.py"]
    out = sys.stdout
    try:
        out.reconfigure(encoding="utf-8")   # Windows'ta ° ve · karakterleri icin
    except Exception:
        pass
    out.write("\033[?1049h\033[?25l\033[2J")
    t0 = time.perf_counter()
    fps = FPS
    try:
        while True:
            a = time.perf_counter()
            size = shutil.get_terminal_size((100, 40))
            W, H = max(40, size.columns - 1), max(16, size.lines - 1)
            t = (a - t0) % CYCLE
            out.write(frame_text(sc, t, W, H, fps))
            out.flush()
            dt = time.perf_counter() - a
            if dt < 1 / FPS:
                time.sleep(1 / FPS - dt)
            fps = int(min(FPS, 1 / max(time.perf_counter() - a, 1e-3)))
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\033[0m\033[?25h\033[?1049l")
        out.flush()


if __name__ == "__main__":
    main()
