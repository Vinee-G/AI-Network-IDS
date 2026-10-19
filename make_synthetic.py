"""Creates SYNTHETIC practice data in the exact NSL-KDD file format (41 features + attack name + difficulty).

  python make_synthetic.py            -> data/synthetic_nsl/KDDTrain+.txt and KDDTest+.txt

The test file contains attack types that never appear in training (apache2, mscan, snmpguess, sqlattack),
like the real KDDTest+. To use the real dataset, just put the real files in another folder and pass --path.
NOTE: synthetic data is easier and cleaner than real traffic. Never quote its scores as real-world results.
"""
import argparse
import os

import numpy as np
import pandas as pd

from nidslib.schema import NSL_CATEGORICAL, NSL_FEATURES

NUM = [c for c in NSL_FEATURES if c not in NSL_CATEGORICAL]


def rec(**kw):
    r = {c: 0 for c in NUM}
    r.update(protocol_type="tcp", service="http", flag="SF")
    for k in kw:
        assert k in r, f"unknown field {k}"
    r.update(kw)
    return r


def B(rng, a, b):
    return round(float(rng.beta(a, b)), 2)


def I(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def LN(rng, mu, sd):
    return int(rng.lognormal(mu, sd))


def pick(rng, items, p=None):
    return str(rng.choice(items, p=p))


# ------------------------------------------------------------- normal traffic
def normal(rng):
    p = pick(rng, ["tcp", "udp", "icmp"], [.82, .14, .04])
    if p == "tcp":
        svc = pick(rng, ["http", "smtp", "ftp_data", "ftp", "telnet", "pop_3", "private", "other"],
                   [.55, .12, .08, .05, .03, .05, .05, .07])
    elif p == "udp":
        svc = pick(rng, ["domain_u", "ntp_u", "other"], [.8, .1, .1])
    else:
        svc = pick(rng, ["eco_i", "ecr_i"], [.7, .3])
    flag = pick(rng, ["SF", "REJ", "S0", "RSTO", "S1"], [.92, .04, .02, .01, .01])
    cnt = int(min(511, 1 + rng.gamma(1.5, 6)))
    srv = int(max(1, min(511, cnt * rng.uniform(.6, 1.4))))
    serr = B(rng, 8, 2) if flag == "S0" else B(rng, .3, 25)
    rerr = B(rng, 8, 2) if flag == "REJ" else B(rng, .3, 25)
    return rec(
        duration=0 if rng.random() < .8 else int(rng.exponential(200)),
        protocol_type=p, service=svc, flag=flag,
        src_bytes=LN(rng, 6, 1.8) if p != "icmp" else int(rng.choice([8, 18, 28])),
        dst_bytes=LN(rng, 7, 2) if p == "tcp" else (LN(rng, 4, 1) if p == "udp" else 0),
        logged_in=int(p == "tcp" and rng.random() < .7),
        hot=I(rng, 1, 3) if (svc.startswith("ftp") and rng.random() < .1) else 0,
        count=cnt, srv_count=srv, serror_rate=serr, srv_serror_rate=serr, rerror_rate=rerr, srv_rerror_rate=rerr,
        same_srv_rate=B(rng, 8, 1), diff_srv_rate=B(rng, 1, 15), srv_diff_host_rate=B(rng, 1, 10),
        dst_host_count=I(rng, 30, 255), dst_host_srv_count=I(rng, 10, 255),
        dst_host_same_srv_rate=B(rng, 6, 1.5), dst_host_diff_srv_rate=B(rng, 1, 10),
        dst_host_same_src_port_rate=B(rng, 1, 6), dst_host_srv_diff_host_rate=B(rng, 1, 10),
        dst_host_serror_rate=serr, dst_host_srv_serror_rate=serr, dst_host_rerror_rate=rerr,
        dst_host_srv_rerror_rate=rerr)


def benign_odd(rng):
    """Legitimate traffic that LOOKS suspicious (busy server, admin ping sweep, admin root session)."""
    r, k = normal(rng), rng.random()
    if k < .4:
        r.update(count=I(rng, 100, 300), srv_count=I(rng, 100, 300), same_srv_rate=B(rng, 15, 1))
    elif k < .7:
        r.update(protocol_type="icmp", service="eco_i", flag="SF", src_bytes=8, count=I(rng, 1, 8),
                 srv_diff_host_rate=B(rng, 6, 2), dst_host_srv_diff_host_rate=B(rng, 6, 2))
    else:
        r.update(protocol_type="tcp", service="telnet", flag="SF", logged_in=1, root_shell=1,
                 num_root=I(rng, 1, 4), num_shells=1, duration=I(rng, 30, 600))
    return r


# ------------------------------------------------------------- DoS
def neptune(rng):
    return rec(service=pick(rng, ["private", "other", "http", "telnet", "ftp"], [.5, .2, .15, .1, .05]),
               flag=pick(rng, ["S0", "REJ", "RSTO"], [.85, .12, .03]), count=I(rng, 120, 511),
               srv_count=I(rng, 1, 200), serror_rate=B(rng, 20, 1), srv_serror_rate=B(rng, 20, 1),
               same_srv_rate=float(rng.choice([B(rng, 1, 12), B(rng, 12, 1)])), diff_srv_rate=B(rng, 1, 12),
               dst_host_count=255, dst_host_srv_count=I(rng, 1, 40), dst_host_same_srv_rate=B(rng, 1, 10),
               dst_host_serror_rate=B(rng, 25, 1), dst_host_srv_serror_rate=B(rng, 25, 1))


def smurf(rng):
    return rec(protocol_type="icmp", service="ecr_i", src_bytes=int(rng.choice([520, 1032], p=[.3, .7])),
               count=511, srv_count=511, same_srv_rate=1.0, dst_host_count=255, dst_host_srv_count=255,
               dst_host_same_srv_rate=1.0, dst_host_same_src_port_rate=1.0)


def back(rng):
    return rec(src_bytes=int(rng.normal(54540, 500)), dst_bytes=int(rng.normal(8314, 1000)), logged_in=1,
               hot=2, num_compromised=I(rng, 1, 2), count=I(rng, 2, 8), srv_count=I(rng, 2, 8),
               same_srv_rate=1.0, dst_host_count=I(rng, 1, 200), dst_host_srv_count=I(rng, 1, 200),
               dst_host_same_srv_rate=B(rng, 10, 1))


def teardrop(rng):
    return rec(protocol_type="udp", service="other", src_bytes=28, wrong_fragment=3, count=I(rng, 50, 150),
               srv_count=I(rng, 50, 150), same_srv_rate=1.0, dst_host_count=255, dst_host_srv_count=I(rng, 20, 120))


def apache2(rng):  # test-only: low serror rate, HTTP flood with large requests
    return rec(flag=pick(rng, ["SF", "S1"], [.7, .3]), src_bytes=LN(rng, 9.5, .7), dst_bytes=LN(rng, 8, 1),
               count=I(rng, 100, 300), srv_count=I(rng, 100, 300), same_srv_rate=B(rng, 15, 1),
               serror_rate=B(rng, .3, 20), dst_host_count=255, dst_host_srv_count=255)


# ------------------------------------------------------------- Probe
def ipsweep(rng):
    return rec(protocol_type="icmp", service="eco_i", src_bytes=int(rng.choice([8, 18])), count=I(rng, 1, 6),
               srv_count=I(rng, 1, 4), srv_diff_host_rate=B(rng, 10, 1), dst_host_count=I(rng, 1, 255),
               dst_host_srv_count=I(rng, 1, 30), dst_host_srv_diff_host_rate=B(rng, 10, 1))


def portsweep(rng):
    fl = pick(rng, ["REJ", "RSTO", "S0", "SF"], [.45, .25, .1, .2])
    return rec(service=pick(rng, ["private", "other", "http", "ftp", "telnet", "smtp"]), flag=fl,
               count=I(rng, 1, 12), srv_count=I(rng, 1, 3), rerror_rate=B(rng, 8, 2) if fl == "REJ" else B(rng, 1, 20),
               diff_srv_rate=B(rng, 8, 2), same_srv_rate=B(rng, 1, 6), dst_host_count=I(rng, 1, 100),
               dst_host_diff_srv_rate=B(rng, 6, 2), dst_host_same_srv_rate=B(rng, 1, 6))


def nmap(rng):
    p = pick(rng, ["tcp", "udp", "icmp"], [.5, .3, .2])
    svc = {"tcp": "private", "udp": "other", "icmp": "ecr_i"}[p]
    return rec(protocol_type=p, service=svc, flag=pick(rng, ["SF", "RSTO", "S0"], [.5, .3, .2]),
               src_bytes=int(rng.choice([0, 8, 18, 28])), count=I(rng, 1, 10), srv_count=I(rng, 1, 5),
               diff_srv_rate=B(rng, 3, 3), dst_host_diff_srv_rate=B(rng, 4, 3), dst_host_count=I(rng, 1, 255))


def mscan(rng):  # test-only: noisy scan against many services
    return rec(service=pick(rng, ["http", "ftp", "telnet", "smtp", "pop_3"]),
               flag=pick(rng, ["SF", "REJ", "RSTO"], [.5, .3, .2]), src_bytes=I(rng, 30, 300), dst_bytes=I(rng, 0, 2000),
               count=I(rng, 150, 511), srv_count=I(rng, 20, 150), diff_srv_rate=B(rng, 4, 4), rerror_rate=B(rng, 4, 5),
               dst_host_count=255, dst_host_diff_srv_rate=B(rng, 4, 4), dst_host_rerror_rate=B(rng, 4, 5))


# ------------------------------------------------------------- R2L
def guess_passwd(rng):
    return rec(service=pick(rng, ["telnet", "ftp", "pop_3", "imap4"]), duration=I(rng, 0, 10),
               src_bytes=LN(rng, 4.8, .3), dst_bytes=LN(rng, 5.3, .4),
               num_failed_logins=I(rng, 1, 5) if rng.random() < .7 else 0, logged_in=int(rng.random() < .3),
               count=I(rng, 1, 5), srv_count=I(rng, 1, 5))


def warezclient(rng):
    return rec(service="ftp_data", duration=I(rng, 0, 60), src_bytes=LN(rng, 8, 1.2), logged_in=1,
               is_guest_login=1, hot=I(rng, 1, 5), count=I(rng, 1, 4), srv_count=I(rng, 1, 4))


def ftp_write(rng):
    return rec(service="ftp", logged_in=1, hot=I(rng, 1, 4), num_file_creations=I(rng, 1, 3),
               num_access_files=I(rng, 0, 3), src_bytes=LN(rng, 6, .6), dst_bytes=LN(rng, 6, .6), is_guest_login=1)


def snmpguess(rng):  # test-only
    return rec(protocol_type="udp", service="private", src_bytes=LN(rng, 4.5, .2), dst_bytes=LN(rng, 4.6, .3),
               count=I(rng, 10, 80), srv_count=I(rng, 10, 80), same_srv_rate=1.0)


# ------------------------------------------------------------- U2R
def buffer_overflow(rng):
    return rec(service="telnet", duration=I(rng, 10, 300), src_bytes=LN(rng, 7, .6), dst_bytes=LN(rng, 8, .8),
               logged_in=1, hot=I(rng, 0, 4), num_compromised=I(rng, 1, 6), root_shell=int(rng.random() < .6),
               num_root=I(rng, 0, 8), num_file_creations=I(rng, 0, 3), num_shells=I(rng, 0, 2),
               num_access_files=I(rng, 0, 2), count=I(rng, 1, 3))


def rootkit(rng):
    return rec(service=pick(rng, ["telnet", "ftp_data", "other"]), logged_in=1, num_root=I(rng, 0, 30),
               num_file_creations=I(rng, 0, 12), root_shell=int(rng.random() < .3), hot=I(rng, 0, 6),
               num_compromised=I(rng, 0, 10), duration=I(rng, 0, 200), count=I(rng, 1, 3))


def loadmodule(rng):
    return rec(service="telnet", logged_in=1, root_shell=1, num_root=I(rng, 0, 3), num_file_creations=I(rng, 0, 3),
               num_shells=int(rng.random() < .5), hot=I(rng, 0, 3), src_bytes=LN(rng, 6.5, .4), duration=I(rng, 0, 100))


def sqlattack(rng):  # test-only
    return rec(logged_in=1, root_shell=1, su_attempted=1, num_root=I(rng, 1, 4), num_file_creations=I(rng, 0, 2),
               hot=I(rng, 1, 3), src_bytes=LN(rng, 7, .5))


SUB = {"DoS": dict(neptune=neptune, smurf=smurf, back=back, teardrop=teardrop, apache2=apache2),
       "Probe": dict(ipsweep=ipsweep, portsweep=portsweep, nmap=nmap, mscan=mscan),
       "R2L": dict(guess_passwd=guess_passwd, warezclient=warezclient, ftp_write=ftp_write, snmpguess=snmpguess),
       "U2R": dict(buffer_overflow=buffer_overflow, rootkit=rootkit, loadmodule=loadmodule, sqlattack=sqlattack)}

TRAIN_MIX = {"Normal": .52, "DoS": .30, "Probe": .11, "R2L": .05, "U2R": .02}
TRAIN_SUB = {"DoS": dict(neptune=.55, smurf=.30, back=.10, teardrop=.05),
             "Probe": dict(ipsweep=.35, portsweep=.35, nmap=.30),
             "R2L": dict(guess_passwd=.4, warezclient=.4, ftp_write=.2),
             "U2R": dict(buffer_overflow=.4, rootkit=.3, loadmodule=.3)}
TEST_MIX = {"Normal": .45, "DoS": .30, "Probe": .13, "R2L": .08, "U2R": .04}
TEST_SUB = {"DoS": dict(neptune=.35, smurf=.25, back=.15, teardrop=.10, apache2=.15),
            "Probe": dict(ipsweep=.25, portsweep=.25, nmap=.20, mscan=.30),
            "R2L": dict(guess_passwd=.25, warezclient=.25, ftp_write=.15, snmpguess=.35),
            "U2R": dict(buffer_overflow=.25, rootkit=.20, loadmodule=.25, sqlattack=.30)}
STEALTH = {"DoS": .04, "Probe": .10, "R2L": .20, "U2R": .20}  # low-and-slow attacks that look like normal traffic
BLUR = {"DoS": .22, "Probe": .30, "R2L": .42, "U2R": .48}  # chance a feature looks 'normal' (overlap)


def generate(n, mix, sub, seed):
    rng = np.random.default_rng(seed)
    cats = list(mix)
    p = np.array(list(mix.values()))
    p = p / p.sum()
    rows = []
    for cat in rng.choice(cats, size=n, p=p):
        if cat == "Normal":
            r, name = (benign_odd(rng) if rng.random() < .12 else normal(rng)), "normal"
        else:
            names = list(sub[cat])
            sp = np.array(list(sub[cat].values()))
            name = str(rng.choice(names, p=sp / sp.sum()))
            r = SUB[cat][name](rng)
            other = normal(rng)
            if rng.random() < STEALTH[cat]:  # keep only protocol/service of the attack; the rest looks normal
                other.update(protocol_type=r["protocol_type"], service=r["service"])
                r = other
            else:
                for k in NUM:
                    if rng.random() < BLUR[cat]:
                        r[k] = other[k]
        r["attack_name"] = name
        rows.append(r)
    df = pd.DataFrame(rows)[NSL_FEATURES + ["attack_name"]]
    df["difficulty"] = 21
    return df


def main(out="data/synthetic_nsl", n_train=30000, n_test=8000, seed=42):
    os.makedirs(out, exist_ok=True)
    generate(n_train, TRAIN_MIX, TRAIN_SUB, seed).to_csv(os.path.join(out, "KDDTrain+.txt"), header=False, index=False)
    generate(n_test, TEST_MIX, TEST_SUB, seed + 1).to_csv(os.path.join(out, "KDDTest+.txt"), header=False, index=False)
    print(f"created {out}/KDDTrain+.txt ({n_train} rows) and KDDTest+.txt ({n_test} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_nsl")
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-test", type=int, default=8000)
    a = ap.parse_args()
    main(a.out, a.n_train, a.n_test)
