#!/usr/bin/env python3
"""
make_synthetic_data.py — generate a small synthetic dataset with the same layout
as the challenge (train: US + India, test: US + India + France) so that
ber_pipeline.py can be smoke-tested end-to-end without the real files.

    python make_synthetic_data.py --out dataset --n-train 3000 --n-test 2000

The noise model is deliberately nasty: abbreviations, legal-suffix swaps,
typos, dropped address components, landmark references, word swaps, and
hard-negative distractors (different business, same street / same name stem).
"""
import argparse, os, random, string
import pandas as pd

WORDS = {
    "US": ["Summit", "Pioneer", "Harbor", "Maple", "Atlas", "Liberty", "Crest", "Falcon", "Granite", "Beacon",
           "Cedar", "Meridian", "Horizon", "Keystone", "Silver", "Oak", "Northwind", "Riverbend", "Copper", "Evergreen"],
    "India": ["Sharma", "Verma", "Gupta", "Shree", "Krishna", "Lakshmi", "Ganesh", "Bharat", "Rajdhani", "Balaji",
              "Om", "Sai", "Annapurna", "Jai", "Mahalaxmi", "Patel", "Agarwal", "Nataraj", "Kaveri", "Sunrise"],
    "France": ["Lumière", "Château", "Rivière", "Étoile", "Boulanger", "Provence", "Marais", "Soleil", "Belle", "Forêt",
               "Atelier", "Maison", "Jardin", "Côte", "Pâtisserie", "Dupont", "Martin", "Bernard", "Lefèvre", "Moreau"],
}
KINDS = {
    "US": ["Traders", "Logistics", "Supply", "Motors", "Dental", "Bakery", "Consulting", "Hardware", "Foods", "Auto Repair"],
    "India": ["Traders", "Enterprises", "Textiles", "Sweets", "Electricals", "Steels", "Agencies", "Medicals", "Foods", "Motors"],
    "France": ["Boulangerie", "Traiteur", "Conseil", "Transports", "Immobilier", "Bijouterie", "Garage", "Épicerie", "Librairie", "Pharmacie"],
}
SUFFIX = {
    "US": [("Inc", "Incorporated"), ("Corp", "Corporation"), ("LLC", "L.L.C."), ("Co", "Company"), ("", "")],
    "India": [("Pvt Ltd", "Private Limited"), ("Ltd", "Limited"), ("LLP", "L.L.P."), ("& Sons", "and Sons"), ("", "")],
    "France": [("SARL", "S.A.R.L."), ("SAS", "S.A.S."), ("SA", "Société Anonyme"), ("EURL", "E.U.R.L."), ("", "")],
}
STREET = {
    "US": [("Main", "St", "Street"), ("Oak", "Ave", "Avenue"), ("Lincoln", "Rd", "Road"), ("Park", "Blvd", "Boulevard"), ("Elm", "Dr", "Drive")],
    "India": [("MG", "Rd", "Road"), ("Station", "Rd", "Road"), ("Gandhi", "Nagar", "Nagar"), ("Nehru", "Marg", "Marg"), ("Anna", "Salai", "Salai")],
    "France": [("de la République", "Rue", "Rue"), ("Victor Hugo", "Av", "Avenue"), ("de Gaulle", "Bd", "Boulevard"), ("des Lilas", "Rue", "Rue"), ("Pasteur", "Pl", "Place")],
}
CITY = {"US": [("Springfield", "IL"), ("Austin", "TX"), ("Denver", "CO"), ("Boston", "MA")],
        "India": [("Bengaluru", "Karnataka"), ("Pune", "Maharashtra"), ("Chennai", "Tamil Nadu"), ("Jaipur", "Rajasthan")],
        "France": [("Lyon", ""), ("Paris", ""), ("Marseille", ""), ("Nantes", "")]}
LANDMARK = {"US": ["near Walmart", "next to Shell gas station"], "India": ["Near SBI ATM", "Opp Bus Stand", "Behind Apollo Hospital"],
            "France": ["près de la gare", "face à la mairie"]}


def postal(c, rnd):
    if c == "India":
        return "".join(rnd.choice("123456789") for _ in range(6))
    return "".join(rnd.choice("0123456789") for _ in range(5))


def typo(s, rnd):
    if len(s) < 4:
        return s
    i = rnd.randrange(1, len(s) - 1)
    op = rnd.random()
    if op < 0.4:
        return s[:i] + s[i + 1:]
    if op < 0.8:
        return s[:i] + rnd.choice(string.ascii_lowercase) + s[i:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_entity(c, rnd, eid):
    w = rnd.sample(WORDS[c], 2)
    kind = rnd.choice(KINDS[c])
    suf = rnd.choice(SUFFIX[c])
    st = rnd.choice(STREET[c])
    city = rnd.choice(CITY[c])
    return dict(id=eid, country=c, words=w, kind=kind, suf=suf, num=str(rnd.randint(1, 999)),
                street=st, city=city, postal=postal(c, rnd), landmark=rnd.choice(LANDMARK[c]))


def render_name(e, rnd, noisy):
    words = list(e["words"])
    name = " ".join(words) + " " + e["kind"]
    suf = e["suf"][0]
    if noisy:
        r = rnd.random()
        if r < 0.3:
            suf = e["suf"][1]
        elif r < 0.45:
            suf = ""
        if rnd.random() < 0.15:
            words = words[::-1]
            name = " ".join(words) + " " + e["kind"]
        if rnd.random() < 0.25:
            name = typo(name, rnd)
        if rnd.random() < 0.1:
            name = name.replace(" and ", " & ")
        if rnd.random() < 0.08:
            name = name.upper()
    return (name + " " + suf).strip()


def render_addr(e, rnd, noisy):
    st = e["street"]
    if e["country"] == "France":
        base = f"{e['num']} {st[2] if not noisy or rnd.random() < .5 else st[1]} {st[0]}"
    else:
        base = f"{e['num']} {st[0]} {st[2] if not noisy or rnd.random() < .5 else st[1]}"
    parts = [base]
    if not noisy or rnd.random() < 0.8:
        parts.append(e["city"][0])
    if e["city"][1] and (not noisy or rnd.random() < 0.6):
        parts.append(e["city"][1])
    if not noisy or rnd.random() < 0.7:
        parts.append(e["postal"])
    if noisy and rnd.random() < 0.3:
        parts.insert(1, e["landmark"])
    if noisy and rnd.random() < 0.15:
        parts[0] = parts[0].split(" ", 1)[1]  # drop street number
    if noisy and rnd.random() < 0.2:
        parts = parts[1:] + parts[:1]  # reorder components
    a = ", ".join(parts)
    if noisy and rnd.random() < 0.2:
        a = typo(a, rnd)
    return a


def build(split, n_entities, countries, seed, out):
    rnd = random.Random(seed)
    s1, s2, s3, gt = [], [], [], []
    c2 = c3 = 0
    for i in range(n_entities):
        c = rnd.choice(countries)
        e = make_entity(c, rnd, f"S1-{split[0].upper()}{i:06d}")
        s1.append((e["id"], render_name(e, rnd, False), render_addr(e, rnd, False), c))
        matches = []
        k = rnd.choices([0, 1, 2, 3, 4, 5], weights=[0.3, 0.3, 0.18, 0.12, 0.06, 0.04])[0]
        for _ in range(k):
            if rnd.random() < 0.55:
                mid = f"S2-{split[0].upper()}{c2:06d}"; c2 += 1
                s2.append((mid, render_name(e, rnd, True), render_addr(e, rnd, True), c))
            else:
                mid = f"S3-{split[0].upper()}{c3:06d}"; c3 += 1
                s3.append((mid, render_name(e, rnd, True), render_addr(e, rnd, True), c))
            matches.append(mid)
        gt.append((e["id"], ",".join(matches)))
        # hard-negative distractors: same street or same name stem, different business
        if rnd.random() < 0.5:
            d = make_entity(c, rnd, "")
            if rnd.random() < 0.5:
                d["street"], d["city"], d["postal"] = e["street"], e["city"], e["postal"]
            else:
                d["words"] = [e["words"][0], rnd.choice(WORDS[c])]
            mid = f"S2-{split[0].upper()}{c2:06d}"; c2 += 1
            s2.append((mid, render_name(d, rnd, True), render_addr(d, rnd, True), c))
        if rnd.random() < 0.3:
            d = make_entity(c, rnd, "")
            mid = f"S3-{split[0].upper()}{c3:06d}"; c3 += 1
            s3.append((mid, render_name(d, rnd, True), render_addr(d, rnd, True), c))
    cols = ["entity_id", "business_name", "business_address", "country"]
    d = os.path.join(out, split)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(s1, columns=cols).to_csv(f"{d}/{split}_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2, columns=cols).sample(frac=1, random_state=seed).to_csv(f"{d}/{split}_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3, columns=cols).sample(frac=1, random_state=seed).to_csv(f"{d}/{split}_source3.tsv", sep="\t", index=False)
    if split == "train":
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(f"{d}/train_ground_truth.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(f"{d}/_hidden_test_ground_truth.tsv", sep="\t", index=False)
    print(f"{split}: S1={len(s1)} S2={len(s2)} S3={len(s3)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build("train", a.n_train, ["US", "India"], a.seed, a.out)
    build("test", a.n_test, ["US", "India", "France"], a.seed + 1, a.out)
