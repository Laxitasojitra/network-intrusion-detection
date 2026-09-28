"""Load the NSL-KDD dataset and map raw attack names to the five standard categories."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Byte counts are extremely skewed (0 to over 1 billion), so they get a log transform
SKEWED_FEATURES = ["duration", "src_bytes", "dst_bytes"]

# num_outbound_cmds is 0 in every row, so it carries no information
DROP_FEATURES = ["num_outbound_cmds"]

NUMERIC_FEATURES = [
    c for c in COLUMNS
    if c not in CATEGORICAL_FEATURES + DROP_FEATURES + ["attack", "difficulty"]
]

ATTACK_CATEGORY = {
    "normal": "Normal",
    # Denial of Service: flood the target so it can't serve real users
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop",
                          "apache2", "mailbomb", "processtable", "udpstorm"]},
    # Probe: scan the network to find open ports and weak machines
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"]},
    # Remote to Local: gain access to a machine from outside
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
                          "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
                          "snmpguess", "xlock", "xsnoop", "worm"]},
    # User to Root: a normal user gains admin (root) privileges
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel",
                          "ps", "sqlattack", "xterm"]},
}

CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["category"] = df["attack"].map(ATTACK_CATEGORY)
    missing = df["category"].isna()
    if missing.any():
        raise ValueError(f"Unmapped attack names: {sorted(df.loc[missing, 'attack'].unique())}")
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def load_train_test():
    train, test = load(TRAIN_PATH), load(TEST_PATH)
    # Attack types in the test set that the model never sees during training ("zero-day" attacks)
    unseen = sorted(set(test["attack"]) - set(train["attack"]))
    test["unseen_attack"] = test["attack"].isin(unseen)
    return train, test, unseen


def features(df):
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
